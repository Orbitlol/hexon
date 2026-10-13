"""
Hexon SentencePiece-Compatible BPE Tokenizer with GGUF `llama.cpp` Parity.

Engineered so that `llama.cpp` (`tokenizer.ggml.model = "llama"` / `LLM_VOCAB_TYPE_SPM`)
tokenizes and detokenizes strings on Android/iOS identically to Python:
- Uses SentencePiece `▁` (`\\u2581`) space escaping (`' '` <-> `'▁'`).
- Includes all single printable ASCII characters, `'▁'`, and 256 `<0xXX>` byte
  fallback tokens in the base vocabulary.
- Builds every multi-character token via iterative pairwise BPE merges `(A, B) -> AB`
  with monotonic scores `score = -float(merge_rank)`, guaranteeing that every
  intermediate sub-token exists in `tokenizer.ggml.tokens` for `llama.cpp`'s
  `llm_tokenizer_spm` bigram priority queue.
- Uses standard ChatML (`<|im_start|>`, `<|im_end|>`) so `llama.cpp` and mobile apps
  (PocketPal, ChatterUI, Maid, Layla) automatically detect the chat template and stop token.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


SPIECE_UNDERLINE = "\u2581"  # "▁" (0xE2 0x96 0x81)

SPECIAL_TOKENS: List[str] = [
    "<|pad|>",        # 0
    "<|bos|>",        # 1
    "<|im_end|>",     # 2 (EOS token — natively recognized by llama.cpp ChatML)
    "<|unk|>",        # 3
    "<|im_start|>",   # 4 (ChatML start token)
    "<|system|>",     # 5
    "<|user|>",       # 6
    "<|assistant|>",  # 7
    "<|eos|>",        # 8
    "<|endoftext|>",  # 9
]

CHATML_TEMPLATE = (
    "{% for message in messages %}"
    "{% if loop.last or message['role'] == 'system' %}"
    "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|im_start|>assistant\n' }}"
    "{% endif %}"
)

SPECIAL_SPLIT_RE = re.compile(
    r"(<\|im_start\|>|<\|im_end\|>|<\|pad\|>|<\|bos\|>|<\|unk\|>|<\|system\|>|<\|user\|>|<\|assistant\|>|<\|eos\|>|<\|endoftext\|>)"
)

# Pre-tokenize normal text into words (with optional leading `▁`), numbers, or punctuation runs
WORD_PRETOKENIZE_RE = re.compile(
    r"(\u2581?[A-Za-z]+|\u2581?[0-9]+|\u2581?[^\u2581A-Za-z0-9\r\n\t]+|[\r\n\t]+|\u2581+)"
)

GPT2_SMOLLM_PAT = re.compile(
    r"'s|'t|'re|'ve|'m|'ll|'d| ?[a-zA-Z]+| ?\d| ?[^\s\da-zA-Z]+|\s+(?!\S)|\s+"
)


def _build_gpt2_byte_maps() -> Tuple[Dict[int, str], Dict[str, int]]:
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    byte_encoder = dict(zip(bs, [chr(c) for c in cs]))
    byte_decoder = {v: k for k, v in byte_encoder.items()}
    return byte_encoder, byte_decoder


_GPT2_BYTE_ENCODER, _GPT2_BYTE_DECODER = _build_gpt2_byte_maps()


class HexonTokenizer:
    """
    SentencePiece-compatible (`llama`) and GPT-2 byte-level (`gpt2`) BPE tokenizer
    whose vocabulary and merge rules execute identically in Python and `llama.cpp`.
    """

    def __init__(
        self,
        tokens: List[str] | None = None,
        scores: List[float] | None = None,
        vocab_size: int = 1024,
        merges: List[str] | None = None,
        tokenizer_model: str = "llama",
        pre_tokenizer: str = "default",
        add_bos_token: bool | None = None,
    ) -> None:
        self.target_vocab_size = vocab_size
        self.tokenizer_model = tokenizer_model
        self.pre_tokenizer = pre_tokenizer
        self.merges: List[str] = list(merges) if merges is not None else []
        if tokens is None:
            tokens, scores = self._build_base_vocab(vocab_size)
        self.tokens: List[str] = list(tokens)
        self.scores: List[float] = (
            list(scores) if scores is not None else [0.0] * len(self.tokens)
        )
        self.add_bos_token: bool = (
            (tokenizer_model == "llama") if add_bos_token is None else bool(add_bos_token)
        )
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        self.token_to_id: Dict[str, int] = {tok: idx for idx, tok in enumerate(self.tokens)}
        self.id_to_token: Dict[int, str] = {idx: tok for idx, tok in enumerate(self.tokens)}
        self.token_to_score: Dict[str, float] = {
            tok: float(self.scores[idx]) for idx, tok in enumerate(self.tokens)
        }
        self.special_token_set = set(SPECIAL_TOKENS)
        self.merge_ranks: Dict[Tuple[str, str], int] = {}
        for rank, m in enumerate(self.merges):
            parts = m.split(" ")
            if len(parts) == 2:
                self.merge_ranks[(parts[0], parts[1])] = rank
        self._bpe_cache: Dict[str, List[int]] = {}

    @property
    def pad_token_id(self) -> int:
        if "<|pad|>" in self.token_to_id:
            return self.token_to_id["<|pad|>"]
        return self.token_to_id.get("<|im_end|>", 2)

    @property
    def bos_token_id(self) -> int:
        if "<|bos|>" in self.token_to_id:
            return self.token_to_id["<|bos|>"]
        return self.token_to_id.get("<|im_start|>", 1)

    @property
    def eos_token_id(self) -> int:
        return self.token_to_id.get("<|im_end|>", 2)

    @property
    def unk_token_id(self) -> int:
        if "<|unk|>" in self.token_to_id:
            return self.token_to_id["<|unk|>"]
        return self.token_to_id.get("<|endoftext|>", 0)

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    def __len__(self) -> int:
        return len(self.tokens)

    @staticmethod
    def _init_atomic_tokens() -> Tuple[List[str], List[float]]:
        tokens: List[str] = list(SPECIAL_TOKENS)
        scores: List[float] = [0.0] * len(SPECIAL_TOKENS)
        # Give ChatML boundary tokens highest BPE merge priority so even
        # `llama_tokenize(..., parse_special=false)` merges them into IDs 2 and 4.
        scores[2] = 1000.0  # <|im_end|>
        scores[4] = 1000.0  # <|im_start|>
        seen = set(tokens)

        # 256 byte fallback tokens `<0x00>` .. `<0xFF>`
        for b in range(256):
            bt = f"<0x{b:02X}>"
            tokens.append(bt)
            scores.append(-10000.0)
            seen.add(bt)

        # Single-character tokens so llama.cpp SPM character split always finds base symbols
        base_chars = [SPIECE_UNDERLINE, "\n", "\r", "\t"] + [
            chr(c) for c in range(33, 127)
        ]
        for ch in base_chars:
            if ch not in seen:
                tokens.append(ch)
                scores.append(-5000.0)
                seen.add(ch)

        # Explicit BPE merge chains for `<|im_start|>` and `<|im_end|>` so
        # `llm_tokenizer_spm_session` merges them even if `parse_special=false`
        chatml_prefixes = [
            "<|", "<|i", "<|im", "<|im_",
            "<|im_s", "<|im_st", "<|im_sta", "<|im_star", "<|im_start", "<|im_start|",
            "<|im_e", "<|im_en", "<|im_end", "<|im_end|",
        ]
        for rank_idx, pref in enumerate(chatml_prefixes):
            if pref not in seen:
                tokens.append(pref)
                scores.append(900.0 - float(rank_idx))
                seen.add(pref)

        return tokens, scores

    @classmethod
    def _build_base_vocab(cls, vocab_size: int) -> Tuple[List[str], List[float]]:
        default_corpus = [
            "You are Hexon, a fast, private AI language model running locally on a mobile device.",
            "I am Hexon, a compact AI language model engineered to run locally on mobile devices with low memory and fast inference.",
            "No. Hexon is trained offline on servers and exported as a frozen GGUF or TorchScript file so it never trains on your mobile device.",
            "Hexon uses Grouped-Query Attention (GQA), tied token embeddings, RMSNorm, SwiGLU, and 4-bit or 8-bit GGUF quantization to keep RAM usage tiny.",
        ]
        return cls._learn_bpe_vocab(default_corpus, vocab_size=vocab_size)

    @classmethod
    def _learn_bpe_vocab(
        cls, texts: Sequence[str], vocab_size: int
    ) -> Tuple[List[str], List[float]]:
        tokens, scores = cls._init_atomic_tokens()
        seen = set(tokens)

        # Count word pieces across corpus (with spaces replaced by `▁`)
        word_freqs: Counter[Tuple[str, ...]] = Counter()
        for text in texts:
            parts = SPECIAL_SPLIT_RE.split(text)
            for part in parts:
                if not part or part in seen:
                    continue
                escaped = part.replace(" ", SPIECE_UNDERLINE)
                for word in WORD_PRETOKENIZE_RE.findall(escaped):
                    chars = tuple(word)
                    if len(chars) >= 2:
                        word_freqs[chars] += 1

        # Iteratively merge the most frequent adjacent pair (tok_a, tok_b) -> tok_a + tok_b
        merge_rank = 1
        words_list = [[list(w), freq] for w, freq in word_freqs.most_common(1500)]

        while len(tokens) < vocab_size and words_list:
            pair_counts: Counter[Tuple[str, str]] = Counter()
            for sym_seq, freq in words_list:
                if len(sym_seq) < 2:
                    continue
                for i in range(len(sym_seq) - 1):
                    pair_counts[(sym_seq[i], sym_seq[i + 1])] += freq

            if not pair_counts:
                break

            (best_a, best_b), best_freq = pair_counts.most_common(1)[0]
            merged = best_a + best_b

            if merged not in seen:
                tokens.append(merged)
                # Higher score = earlier merge in llama.cpp llm_tokenizer_spm max-heap!
                scores.append(-float(merge_rank))
                seen.add(merged)
                merge_rank += 1

            # Replace occurrences of (best_a, best_b) in words_list
            for item in words_list:
                sym_seq = item[0]
                if len(sym_seq) < 2:
                    continue
                new_seq: List[str] = []
                i = 0
                while i < len(sym_seq):
                    if (
                        i < len(sym_seq) - 1
                        and sym_seq[i] == best_a
                        and sym_seq[i + 1] == best_b
                    ):
                        new_seq.append(merged)
                        i += 2
                    else:
                        new_seq.append(sym_seq[i])
                        i += 1
                item[0] = new_seq

            if best_freq <= 1 and len(tokens) >= vocab_size:
                break

        # Pad remaining slots up to exact vocab_size
        idx = 0
        while len(tokens) < vocab_size:
            cand = f"<|reserved_{idx}|>"
            if cand not in seen:
                tokens.append(cand)
                scores.append(-20000.0)
                seen.add(cand)
            idx += 1

        return tokens[:vocab_size], scores[:vocab_size]

    @classmethod
    def train_from_texts(cls, texts: Sequence[str], vocab_size: int = 1024) -> "HexonTokenizer":
        tokens, scores = cls._learn_bpe_vocab(texts, vocab_size=vocab_size)
        return cls(tokens=tokens, scores=scores, vocab_size=vocab_size)

    def _spm_bpe_segment(self, escaped_segment: str) -> List[int]:
        """
        Run the exact same BPE bigram max-score merge algorithm as
        `llama.cpp`'s `llm_tokenizer_spm::tokenize`.
        """
        if not escaped_segment:
            return []

        symbols: List[str] = list(escaped_segment)
        while len(symbols) >= 2:
            best_idx = -1
            best_score = float("-inf")
            best_merged = ""

            for i in range(len(symbols) - 1):
                cand = symbols[i] + symbols[i + 1]
                sc = self.token_to_score.get(cand)
                if sc is not None and sc > best_score:
                    best_score = sc
                    best_idx = i
                    best_merged = cand

            if best_idx < 0:
                break

            symbols[best_idx : best_idx + 2] = [best_merged]

        ids: List[int] = []
        for sym in symbols:
            tid = self.token_to_id.get(sym)
            if tid is not None:
                ids.append(tid)
            else:
                for b in sym.encode("utf-8"):
                    bt = f"<0x{b:02X}>"
                    ids.append(self.token_to_id.get(bt, self.unk_token_id))
        return ids

    def _gpt2_bpe_word(self, word: str) -> List[int]:
        cached = self._bpe_cache.get(word)
        if cached is not None:
            return list(cached)
        syms = list(word)
        while len(syms) >= 2:
            best_rank = 10**9
            best_idx = -1
            for i in range(len(syms) - 1):
                r = self.merge_ranks.get((syms[i], syms[i + 1]), 10**9)
                if r < best_rank:
                    best_rank = r
                    best_idx = i
            if best_idx < 0:
                break
            syms = syms[:best_idx] + [syms[best_idx] + syms[best_idx + 1]] + syms[best_idx + 2 :]
        ids = [self.token_to_id[s] for s in syms if s in self.token_to_id]
        if len(self._bpe_cache) < 16384:
            self._bpe_cache[word] = ids
        return list(ids)

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
        add_space_prefix: bool = False,
    ) -> List[int]:
        """
        Encode text into token IDs using either SentencePiece `▁` BPE (`llama`)
        or GPT-2 byte-level BPE (`gpt2`). Matches `llama.cpp` tokenization behavior.
        """
        ids: List[int] = []
        if add_bos and self.add_bos_token:
            ids.append(self.bos_token_id)

        if self.tokenizer_model == "gpt2":
            parts = SPECIAL_SPLIT_RE.split(text)
            for part in parts:
                if not part:
                    continue
                if part in self.token_to_id and part.startswith("<|") and part.endswith("|>"):
                    ids.append(self.token_to_id[part])
                    continue
                for m in GPT2_SMOLLM_PAT.findall(part):
                    w = "".join(_GPT2_BYTE_ENCODER[b] for b in m.encode("utf-8"))
                    ids.extend(self._gpt2_bpe_word(w))
            if add_eos:
                ids.append(self.eos_token_id)
            return ids

        parts = SPECIAL_SPLIT_RE.split(text)
        first_normal = True
        for part in parts:
            if not part:
                continue
            if part in self.special_token_set:
                ids.append(self.token_to_id[part])
                continue

            escaped = part.replace(" ", SPIECE_UNDERLINE)
            if add_space_prefix and first_normal and not escaped.startswith(SPIECE_UNDERLINE):
                escaped = SPIECE_UNDERLINE + escaped
            first_normal = False
            ids.extend(self._spm_bpe_segment(escaped))

        if add_eos:
            ids.append(self.eos_token_id)
        return ids

    def decode(self, token_ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        """Decode sequence of token IDs back into a UTF-8 string."""
        if self.tokenizer_model == "gpt2":
            pieces: List[str] = []
            for tid in token_ids:
                idx = int(tid)
                tok = self.id_to_token.get(idx, "")
                if skip_special_tokens and (
                    idx in (0, 1, 2)
                    or tok in self.special_token_set
                    or (tok.startswith("<") and tok.endswith(">"))
                ):
                    continue
                pieces.append(tok)
            joined = "".join(pieces)
            raw_bytes = bytes([_GPT2_BYTE_DECODER[c] for c in joined if c in _GPT2_BYTE_DECODER])
            return raw_bytes.decode("utf-8", errors="replace")

        out_parts: List[str] = []
        pending_bytes = bytearray()

        def flush_bytes() -> None:
            if pending_bytes:
                out_parts.append(pending_bytes.decode("utf-8", errors="replace"))
                pending_bytes.clear()

        for tid in token_ids:
            idx = int(tid)
            tok = self.id_to_token.get(idx, "<|unk|>")
            if skip_special_tokens and (
                tok in self.special_token_set or tok.startswith("<|reserved_")
            ):
                continue
            if len(tok) == 6 and tok.startswith("<0x") and tok.endswith(">"):
                try:
                    b = int(tok[3:5], 16)
                    pending_bytes.append(b)
                    continue
                except ValueError:
                    pass
            flush_bytes()
            out_parts.append(tok.replace(SPIECE_UNDERLINE, " "))

        flush_bytes()
        return "".join(out_parts)

    def apply_chat_template(
        self,
        messages: Sequence[Dict[str, str]],
        add_generation_prompt: bool = True,
    ) -> str:
        """
        Format messages using the standard ChatML format natively supported by
        `llama.cpp` (`<|im_start|>role\\ncontent<|im_end|>\\n`).
        """
        msgs = list(messages)
        if self.tokenizer_model == "gpt2":
            parts_gpt2: List[str] = []
            has_sys = any(m.get("role", "").strip().lower() == "system" for m in msgs)
            if not has_sys:
                parts_gpt2.append(
                    "<|im_start|>system\n"
                    "You are Hexon, a helpful, concise, and intelligent AI assistant running locally on a mobile device. "
                    "Your name is Hexon. You are trained offline on servers and exported to GGUF v3, so you never train on the mobile device."
                    "<|im_end|>\n"
                )
            for msg in msgs[-8:]:
                role = msg.get("role", "user").strip().lower()
                content = msg.get("content", "").strip()
                parts_gpt2.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
            if add_generation_prompt:
                parts_gpt2.append("<|im_start|>assistant\n")
            return "".join(parts_gpt2)

        if len(msgs) > 1:
            sys_msgs = [m for m in msgs[:-1] if m.get("role", "").strip().lower() == "system"]
            window = (sys_msgs[:1] if sys_msgs else []) + [msgs[-1]]
        else:
            window = msgs
        parts: List[str] = []
        for msg in window:
            role = msg.get("role", "user").strip().lower()
            content = msg.get("content", "").strip()
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n")
        return "".join(parts)

    def get_gguf_token_types(self) -> List[int]:
        """
        Return GGUF token_type array per token:
        1 = NORMAL, 3 = CONTROL (special), 4 = USER_DEFINED, 6 = BYTE
        """
        user_defined_chat_tokens = {"<|im_start|>", "<|system|>", "<|user|>", "<|assistant|>"}
        types: List[int] = []
        for idx, tok in enumerate(self.tokens):
            if self.tokenizer_model == "gpt2":
                types.append(3 if idx < 17 else 1)
            elif tok in user_defined_chat_tokens:
                # USER_DEFINED (4) is pre-tokenized by llama.cpp even when parse_special=false
                types.append(4)
            elif tok in self.special_token_set:
                types.append(3)  # CONTROL
            elif len(tok) == 6 and tok.startswith("<0x") and tok.endswith(">"):
                types.append(6)  # BYTE
            elif tok.startswith("<|reserved_"):
                types.append(4)  # USER_DEFINED / UNUSED
            else:
                types.append(1)  # NORMAL
        return types

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_type": "gpt2-bpe" if self.tokenizer_model == "gpt2" else "llama-spm-bpe",
            "tokenizer_model": self.tokenizer_model,
            "pre_tokenizer": self.pre_tokenizer,
            "add_bos_token": self.add_bos_token,
            "vocab_size": len(self.tokens),
            "tokens": self.tokens,
            "scores": self.scores,
            "merges": self.merges,
            "chat_template": CHATML_TEMPLATE,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HexonTokenizer":
        return cls(
            tokens=data["tokens"],
            scores=data.get("scores"),
            vocab_size=data.get("vocab_size", len(data["tokens"])),
            merges=data.get("merges"),
            tokenizer_model=data.get("tokenizer_model", "llama"),
            pre_tokenizer=data.get("pre_tokenizer", "default"),
            add_bos_token=data.get("add_bos_token"),
        )

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "HexonTokenizer":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
