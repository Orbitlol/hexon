"""
Comprehensive Test Suite for Hexon Mobile AI Model (PyTorch + GGUF v3 + TorchScript).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from hexon import (
    GGUFReader,
    HexonConfig,
    HexonForCausalLM,
    HexonMobileEngine,
    HexonTokenizer,
    build_sft_corpus,
    export_to_gguf,
    export_to_mobile_int8_pt,
    export_to_torchscript,
    save_sft_dataset_jsonl,
)
from hexon.export_gguf import quantize_tensor_q4_0, quantize_tensor_q8_0
from hexon.gguf_runtime import dequantize_tensor_q4_0, dequantize_tensor_q8_0
from hexon.sft import HexonSFTDataset


def test_config_presets_and_parameter_count() -> None:
    for preset in ("hexon-nano", "hexon-micro", "hexon-mobile-35m", "hexon-mobile-125m"):
        cfg = HexonConfig.from_preset(preset)
        assert cfg.hidden_size % cfg.num_attention_heads == 0
        assert cfg.num_attention_heads % cfg.num_key_value_heads == 0
        mem = cfg.estimate_mobile_memory_mb(seq_len=256)
        assert mem["q4_0_weights_mb"] < mem["q8_0_weights_mb"] < mem["f16_weights_mb"]

    nano_cfg = HexonConfig.from_preset("hexon-nano", vocab_size=512)
    model = HexonForCausalLM(nano_cfg)
    assert model.num_parameters() == nano_cfg.estimate_parameter_count()
    # Tied embeddings check
    assert model.lm_head.weight.data_ptr() == model.embed_tokens.weight.data_ptr()


def test_mobile_inference_lock_blocks_training() -> None:
    cfg = HexonConfig.from_preset("hexon-nano", vocab_size=512)
    model = HexonForCausalLM(cfg)
    model.freeze_for_mobile_inference()

    assert model.mobile_inference_locked is True
    assert not model.training
    assert all(not p.requires_grad for p in model.parameters())

    # Attempting to enable training mode on mobile must raise RuntimeError
    with pytest.raises(RuntimeError, match="locked for mobile inference-only mode"):
        model.train(True)

    # Attempting to compute training loss on mobile must raise RuntimeError
    dummy_ids = torch.randint(0, 512, (1, 8))
    with pytest.raises(RuntimeError, match="disabled on mobile-locked"):
        model(dummy_ids, labels=dummy_ids)


def test_kv_cache_equivalence() -> None:
    torch.manual_seed(123)
    cfg = HexonConfig.from_preset("hexon-nano", vocab_size=512)
    model = HexonForCausalLM(cfg).freeze_for_mobile_inference()

    seq = torch.randint(0, 512, (1, 12))
    full_logits = model(seq, use_cache=False)["logits"]

    # Step through prefix + last token with KV cache
    prefix_out = model(seq[:, :-1], use_cache=True)
    last_out = model(
        seq[:, -1:], past_key_values=prefix_out["past_key_values"], use_cache=True
    )
    cached_last_logit = last_out["logits"][:, -1, :]

    assert torch.allclose(full_logits[:, -1, :], cached_last_logit, atol=1e-5)


def test_tokenizer_roundtrip_and_special_tokens() -> None:
    tok = HexonTokenizer.train_from_texts(
        ["Hello Hexon on mobile!", "Quantized GGUF inference."], vocab_size=512
    )
    assert len(tok) == 512

    sample = "Hexon runs offline on Android & iOS 🚀! Cost: $0.00."
    ids = tok.encode(sample, add_bos=True, add_eos=True)
    assert ids[0] == tok.bos_token_id
    assert ids[-1] == tok.eos_token_id
    decoded = tok.decode(ids, skip_special_tokens=True)
    assert decoded == sample


def test_q8_0_and_q4_0_quantization_accuracy() -> None:
    rng = np.random.default_rng(42)
    weights = rng.normal(0.0, 0.1, size=(64, 192)).astype(np.float32)

    q8_bytes = quantize_tensor_q8_0(weights)
    w_q8 = dequantize_tensor_q8_0(q8_bytes, weights.shape)
    cos_q8 = float(
        np.dot(weights.ravel(), w_q8.ravel())
        / (np.linalg.norm(weights) * np.linalg.norm(w_q8))
    )
    assert cos_q8 > 0.9998

    q4_bytes = quantize_tensor_q4_0(weights)
    w_q4 = dequantize_tensor_q4_0(q4_bytes, weights.shape)
    cos_q4 = float(
        np.dot(weights.ravel(), w_q4.ravel())
        / (np.linalg.norm(weights) * np.linalg.norm(w_q4))
    )
    assert cos_q4 > 0.990


def test_gguf_export_and_reload_roundtrip(tmp_path: Path) -> None:
    torch.manual_seed(7)
    cfg = HexonConfig.from_preset("hexon-nano", vocab_size=512)
    model = HexonForCausalLM(cfg).freeze_for_mobile_inference()
    tok = HexonTokenizer(vocab_size=512)

    input_ids = torch.randint(0, 512, (1, 10))
    orig_logits = model(input_ids)["logits"]

    for qtype, min_cos in [("f32", 0.99999), ("f16", 0.9999), ("q8_0", 0.999), ("q4_0", 0.97)]:
        gguf_path = tmp_path / f"test-{qtype}.gguf"
        info = export_to_gguf(model, tok, gguf_path, quant_type=qtype)
        assert info["gguf_version"] == 3
        assert gguf_path.exists()

        reader = GGUFReader(gguf_path)
        assert reader.metadata["hexon.mobile_inference_only"] is True
        assert reader.metadata["hexon.trainable_on_device"] is False

        reloaded_model, reloaded_tok = reader.load_model_and_tokenizer()
        assert reloaded_model.mobile_inference_locked is True
        assert len(reloaded_tok) == 512

        reloaded_logits = reloaded_model(input_ids)["logits"]
        cos_sim = F.cosine_similarity(
            orig_logits.flatten(), reloaded_logits.flatten(), dim=0
        ).item()
        assert cos_sim >= min_cos


def test_torchscript_and_int8_mobile_exports(tmp_path: Path) -> None:
    torch.manual_seed(11)
    cfg = HexonConfig.from_preset("hexon-nano", vocab_size=512)
    model = HexonForCausalLM(cfg).freeze_for_mobile_inference()
    tok = HexonTokenizer(vocab_size=512)

    ts_path = tmp_path / "model-ts.pt"
    export_to_torchscript(model, ts_path)
    ts_loaded = torch.jit.load(str(ts_path))

    test_ids = torch.randint(0, 512, (1, 12))
    with torch.inference_mode():
        eager_logits = model(test_ids)["logits"]
        ts_logits = ts_loaded(test_ids)
    assert torch.allclose(eager_logits, ts_logits, atol=1e-4)

    int8_path = tmp_path / "model-int8.pt"
    int8_info = export_to_mobile_int8_pt(model, tok, int8_path)
    assert int8_info["mobile_inference_only"] is True
    assert int8_path.exists()


def test_pretrained_exported_gguf_artifacts_work() -> None:
    q8_path = Path("exports/hexon-nano-q8_0.gguf")
    q4_path = Path("exports/hexon-nano-q4_0.gguf")
    f16_path = Path("exports/hexon-nano-f16.gguf")
    assert q8_path.exists(), "Pretrained Q8_0 GGUF file must exist in exports/"
    assert q4_path.exists(), "Pretrained Q4_0 GGUF file must exist in exports/"
    assert f16_path.exists(), "Pretrained F16 GGUF file must exist in exports/"

    engine_f16 = HexonMobileEngine.from_file(f16_path)
    out_id = engine_f16.reply("who are you", max_new_tokens=64, temperature=0.0)
    assert "Hexon" in out_id["response"]
    assert out_id["mobile_inference_locked"] is True

    out_joke = engine_f16.reply("tell me a joke", max_new_tokens=64, temperature=0.0)
    assert "dark mode" in out_joke["response"] or "bugs" in out_joke["response"]

    out_math = engine_f16.reply("what is 2+2", max_new_tokens=64, temperature=0.0)
    assert "4" in out_math["response"]


def test_sft_corpus_and_prompt_masking(tmp_path: Path) -> None:
    corpus = build_sft_corpus()
    assert len(corpus) > 200

    jsonl_path = tmp_path / "sft.jsonl"
    count = save_sft_dataset_jsonl(jsonl_path)
    assert count > 100
    assert jsonl_path.exists()

    tok = HexonTokenizer.train_from_texts([c["full"] for c in corpus[:30]], vocab_size=512)
    ds = HexonSFTDataset(corpus[:10], tok, max_length=128, mask_prompt=True)
    sample = ds[0]
    # Prompt tokens at the start of `labels` must be masked with `pad_token_id`
    assert sample["labels"][0].item() == tok.pad_token_id
    # Assistant response tokens at the end must NOT be masked
    assert (sample["labels"] != tok.pad_token_id).sum().item() > 3


def test_common_crawl_curation_and_base_pretrained_checkpoint(tmp_path: Path) -> None:
    from hexon.curation import CurationConfig, curate_raw_records, run_curation_pipeline

    raw_samples = [
        {
            "url": "https://example.org/science/photosynthesis",
            "text": (
                "<html><body><nav>Skip to content</nav>"
                "<p>Photosynthesis is the biological process by which green plants and "
                "other organisms convert light energy from the sun into chemical energy "
                "stored in glucose molecules. Contact admin@example.org or 192.168.1.10 for details.</p>"
                "<footer>All rights reserved</footer></body></html>"
            ),
            "source": "test_warc",
            "cc_segment": "CC-MAIN-test",
            "lang": "en",
        },
        # Duplicate record that must be removed by deduplication
        {
            "url": "https://mirror.example.org/science/photosynthesis",
            "text": (
                "Photosynthesis is the biological process by which green plants and "
                "other organisms convert light energy from the sun into chemical energy "
                "stored in glucose molecules. Contact admin@example.org or 192.168.1.10 for details."
            ),
            "source": "test_warc",
            "cc_segment": "CC-MAIN-test",
            "lang": "en",
        },
        # Non-English record that must be filtered out
        {
            "url": "https://es.example.org/hola",
            "text": "Escopete es un municipio de España perteneciente a la provincia de Guadalajara.",
            "source": "test_warc",
            "cc_segment": "CC-MAIN-test",
            "lang": "spa",
        },
    ]

    curated, report = curate_raw_records(raw_samples, config=CurationConfig())
    assert len(curated) == 1
    assert report.rejected_url_or_lang == 1
    assert report.rejected_duplicates == 1
    assert "[EMAIL]" in curated[0].text
    assert "[IP]" in curated[0].text
    assert "Skip to content" not in curated[0].text

    # Verify real Common Crawl snapshot curation & pre-trained base checkpoint exist
    curated_docs, full_report = run_curation_pipeline(
        raw_dir="data/raw_cc",
        output_jsonl=tmp_path / "cc_curated.jsonl",
    )
    assert len(curated_docs) >= 200
    assert full_report.total_clean_words >= 50000
    assert Path("exports/hexon-nano-base-pretrained.pt").exists()
    assert Path("exports/hexon-nano-base-f16.gguf").exists()


def test_natural_prompts_and_multiturn_chat_coherence() -> None:
    engine = HexonMobileEngine.from_file("exports/hexon-nano-q8_0.gguf")

    natural_checks = [
        ("tell me a funny joke please", "dark mode"),
        ("how much is 6 times 7", "42"),
        ("why do leaves look green", "Photosynthesis"),
        ("where is the eiffel tower", "Paris"),
        ("which is better cats or dogs", "Dogs"),
        ("can you recommend a good movie to watch", "Interstellar"),
        ("do you need wifi to work", "offline"),
    ]
    for prompt, expected_keyword in natural_checks:
        out = engine.reply(prompt, max_new_tokens=60, temperature=0.0)
        assert expected_keyword in out["response"], f"Failed on {prompt!r}: {out['response']!r}"

    # Verify concatenated multi-turn ChatML session with system prompt
    tok = engine.tokenizer
    model = engine.model
    raw_ctx = "<|im_start|>system\nYou are a helpful AI assistant.<|im_end|>\n"
    turns = [
        ("Hi!", "Hello! I am Hexon"),
        ("How are you doing?", "Doing great"),
        ("What is your name?", "My name is Hexon"),
        ("Can you tell me a joke?", "Joke time"),
        ("What is 5 + 5?", "five plus five is 10"),
        ("Why is the sky blue?", "Sunlight contains"),
        ("Do you like dogs?", "Animals are amazing"),
        ("Thank you!", "You are very welcome"),
    ]
    for user_msg, expected_prefix in turns:
        raw_ctx += f"<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n"
        ids = tok.encode(raw_ctx, add_bos=True, add_eos=False)
        out_ids = model.generate(
            torch.tensor([ids]),
            max_new_tokens=60,
            temperature=0.0,
            eos_token_id=tok.eos_token_id,
        )[0, len(ids) :].tolist()
        if tok.eos_token_id in out_ids:
            out_ids = out_ids[: out_ids.index(tok.eos_token_id)]
        resp = tok.decode(out_ids).strip()
        assert expected_prefix in resp, f"Turn {user_msg!r} failed: {resp!r}"
        raw_ctx += f"{resp}<|im_end|>\n"


def test_chromadb_rag_pipeline_and_embedding_function(tmp_path: Path) -> None:
    from hexon.rag import HexonChromaEmbeddingFunction, HexonChromaRAG, chunk_text

    ef = HexonChromaEmbeddingFunction(dim=384)
    vecs = ef(["Poco M3 has a Snapdragon 662 CPU", "Photosynthesis happens in chloroplasts"])
    assert len(vecs) == 2
    assert len(vecs[0]) == 384

    pieces = chunk_text("First sentence here. " * 40, chunk_size=200, chunk_overlap=32)
    assert len(pieces) >= 3

    rag = HexonChromaRAG(persist_dir=tmp_path / "chroma_test", auto_ingest_defaults=True)
    assert rag.count() >= 100

    hits = rag.retrieve("What processor and battery does the Poco M3 have?", top_k=2)
    assert len(hits) >= 1
    assert "Snapdragon 662" in hits[0].text

    aug_prompt, chunks = rag.build_rag_prompt("How does ChromaDB HNSW vector search work?", top_k=2)
    assert "Context from ChromaDB:" in aug_prompt
    assert len(chunks) >= 1


def test_hexon_135m_compositional_and_rag_inference() -> None:
    gguf_135m = Path("exports/hexon-135m-q4_0.gguf")
    assert gguf_135m.exists(), "135M GGUF model (50-100 MB range) must exist in exports/"

    reader = GGUFReader(gguf_135m)
    summary = reader.summary()
    assert 50.0 <= summary["file_size_mb"] <= 100.0
    assert summary["mobile_inference_only"] is True
    assert summary["trainable_on_device"] is False
    assert summary["num_layers"] == 30
    assert summary["hidden_size"] == 576

    engine = HexonMobileEngine.from_file(gguf_135m)
    res_rag = engine.reply(
        "What processor and battery does the Poco M3 have?",
        use_rag=True,
        max_new_tokens=48,
        temperature=0.0,
    )
    assert res_rag["rag_enabled"] is True
    assert len(res_rag["retrieved_chunks"]) >= 1
    assert "Snapdragon 662" in res_rag["response"] or "6000" in res_rag["response"]



