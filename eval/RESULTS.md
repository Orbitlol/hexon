# Hexon evaluation results

All numbers below come from `scripts/run_eval.py` scoring the committed
benchmark in `eval/heldout_eval.json`. Raw responses for every item are stored
next to these numbers in `eval/results/` (old knowledge base) and
`eval/results_v2/` (rebuilt knowledge base), so anything here can be audited
item by item without re-running a model.

## What is being measured

| Set | Items | What it is |
| :--- | :---: | :--- |
| `heldout_v1` | 50 | Fresh questions verified absent from every training corpus. **This is the only set that supports model-vs-model comparison.** |
| `legacy_keyword_check` | 8 | The original Poco M3 / photosynthesis / sky / prime / binary search / linked list check. **Contaminated — see below.** |
| `multiturn_anti_copy` | 3 | Two-turn sequences checking a later turn does not repeat an earlier one. |

Metrics:

*   **acc** — every `must_include` alias group matched (AND across groups, OR
    within a group). Keyword-based, so it is a floor on quality, not a grade.
*   **hall** — a `must_not` pattern fired outside a caveat window (see
    "Scoring bug found and fixed" below).
*   **acc_g / acc_u** — accuracy split by whether the retrieved ChromaDB chunk
    already contained the answer. `acc_u` is the honest generalisation number
    once the knowledge base covers the test domains.

## The headline comparison

Held-out set, RAG on, rebuilt knowledge base (the configuration that ships):

| Model | acc | hall | grounded | acc_g | acc_u |
| :--- | :---: | :---: | :---: | :---: | :---: |
| hexon-135m-q4_0 | 0.720 | 0.000 | 34/50 | 0.735 | 0.688 |
| hexon-code | 0.720 | 0.000 | 34/50 | 0.765 | 0.625 |
| hexon-1.2 | 0.660 | 0.000 | 34/50 | 0.676 | 0.625 |
| **hexon-1.5** | **0.760** | 0.040 | 34/50 | **0.853** | 0.562 |

## Effect of the knowledge base rebuild

Same three models, RAG on, before and after rebuilding ChromaDB:

| Model | old KB acc | new KB acc | delta | old KB hall | new KB hall |
| :--- | :---: | :---: | :---: | :---: | :---: |
| hexon-135m-q4_0 | 0.580 | 0.720 | **+0.140** | 0.040 | 0.000 |
| hexon-code | 0.520 | 0.720 | **+0.200** | 0.020 | 0.000 |
| hexon-1.2 | 0.500 | 0.660 | **+0.160** | 0.040 | 0.000 |

Before the rebuild, 146 of 225 chunks (67%) came from raw Common Crawl web
scrapes. Sampled content included invoice-factoring teleconference spam and
9/11 conspiracy blog posts. Retrieval was returning that instead of answers,
which is why **turning RAG on made every model worse than running bare**:

| Model | RAG off | RAG on (old KB) |
| :--- | :---: | :---: |
| hexon-135m-q4_0 | 0.640 | 0.580 |
| hexon-code | 0.680 | 0.520 |
| hexon-1.2 | 0.640 | 0.500 |

Two changes produced the improvement: replacing the 146 web-scrape chunks with
149 curated reference chunks, and raising the retrieval embedding width from
384 to 2048 (`DEFAULT_EMBEDDING_DIM` in `hexon/rag.py`). Measured recall@1 on
88 labelled question/chunk pairs rose from 0.318 to 0.409; with the lexical
reranker, top-1 is 0.545 and top-3 is 0.830.

## The legacy keyword check is not a benchmark

`scripts/run_eval.py` runs a Jaccard contamination check against every training
corpus on every run. It reports that **4 of the 8 legacy items appear verbatim
in training data**:

```
legacy-04 "What is photosynthesis?"         jaccard=1.0  exports/hexon-nano-sft-dataset.jsonl
legacy-05 "Why is the sky blue?"            jaccard=1.0  exports/hexon-nano-sft-dataset.jsonl
legacy-06 "What is a prime number?"         jaccard=1.0  data/curated/alpaca_knowledge.jsonl
legacy-08 "How do you reverse a linked list?" jaccard=1.0 data/curated/alpaca_knowledge.jsonl
```

All 8 items were additionally used in hexon-1.2's 14-example fine-tune. This
matters because hexon-1.2's headline result was 8/8 on this check. On the
contaminated set it still scores joint-highest (0.750); on held-out questions
it scores **lowest** of the four models (0.660).

| Model | legacy (contaminated) | held-out (clean) |
| :--- | :---: | :---: |
| hexon-135m-q4_0 | 0.750 | 0.720 |
| hexon-code | 0.500 | 0.720 |
| hexon-1.2 | **0.750** | 0.660 |
| hexon-1.5 | 0.500 | 0.760 |

The 8/8 was measuring memorisation of the test set. It is retained here only
for continuity with older runs and should not be quoted as a quality signal.

## Hallucinations

Honest status: **the ~30% hallucination reduction target has not been met as a
model improvement.**

What did reduce hallucination flags was the knowledge base, not the model:
RAG-on flags fell from 0.020–0.040 to 0.000 for all three pre-existing models.

hexon-1.5 flags 2 of 50 items, which is *higher* than the 0–1 the older models
flag. Reviewing both by hand:

*   **h-019, "Which planet is closest to the Sun?" — genuine hallucination.**
    It answered about Venus being the hottest planet (867 °F, thick CO₂
    atmosphere) instead of Mercury. This is a real failure.
*   **h-014, "In which year did the Berlin Wall fall?" — correct answer,
    garbled continuation.** It gives 1989 correctly, then correctly notes the
    USSR dissolved in 1991, then trails off into a repeated, self-contradictory
    sentence ("The Soviet Union opened the Berlin Wall two years later…").
    The flagged `1991` is legitimate context; the tail is a real defect.

Two further caveats on these numbers. With 50 items, one flag is 0.020, so the
gap between 0.000 and 0.040 is **within noise** and no ranking should be
inferred from it. And the metric only catches wrong answers that were
anticipated in `must_not` — it cannot detect a confident, plausible fabrication
on a topic nobody thought to write a pattern for. Treat it as a lower bound.

The `unsupported_number` probe (a number in the response that appears in
neither the question nor the reference) is reported separately per run and is
also noisy: hexon-1.5 shows 3 such items.

## Epoch selection

Training loss on 126 examples falls to 0.28 by epoch 5 and 0.018 by epoch 8,
which looks like memorisation. It is worth checking whether an earlier, less
fitted checkpoint generalises better, so epoch 5 was exported and scored the
same way (`scripts/export_checkpoint.py`). It is worse on both counts:

| Checkpoint | train loss | held-out acc | hall |
| :--- | :---: | :---: | :---: |
| epoch 5 | 0.279 | 0.640 | 0.120 |
| **epoch 8 (shipped)** | 0.018 | **0.760** | **0.040** |

So the extra epochs helped rather than hurt, and the shipped model is epoch 8.
This is the opposite of what the training curve suggests, which is the argument
for measuring instead of assuming.

## Multi-turn anti-copying

| Model | copy rate | expectation met |
| :--- | :---: | :---: |
| hexon-135m-q4_0 | 0.000 | 1.000 |
| hexon-code | 0.000 | 1.000 |
| hexon-1.2 | 0.333 | 0.333 |
| hexon-1.5 | 0.000 | 1.000 |

hexon-1.2 regressed here once the knowledge base changed: on one of three
sequences it repeated its previous turn. hexon-1.5 was trained on 8 multi-turn
sequences with distinct answers per turn and shows no copying.

## Scoring bug found and fixed

The first version of the scorer flagged any `must_not` hit as a hallucination.
That punished correct contrastive answers, because careful answers name the
wrong option in order to rule it out. Real examples that were being counted as
hallucinations:

*   "Ottawa, which sits on the Ontario side of the border with Quebec. **It is
    not** Toronto or Montreal." — flagged for `toronto`, `montreal`
*   "**The yen**… should **not be confused with** the yuan of China or the won
    of South Korea." — flagged for `yuan`, `won`
*   "**The Sahara** in North Africa. Antarctica is **technically** the largest
    desert overall if you define desert by low precipitation **rather than** by
    heat." — flagged for `antarctic`

hexon-1.5 was hit hardest by this because its training answers deliberately use
that contrastive style, which is what pushed its apparent rate to 0.180.

Two fixes: `hexon/evalbench.py` now discounts a `must_not` hit when a caveat
cue (`not`, `rather than`, `technically`, `disputed`, `confused with`, …)
appears within 90 characters on either side of the match — the window is
symmetric because caveats often follow the mention. Two broken patterns were
also removed from the eval set: `\bin\b` (which matched the English
preposition "in", so almost any sentence tripped it) and `\b15\b` on "15% of
200" (which matched a correct answer restating the question).

`scripts/rescore.py` re-applies the current scorer to every saved response, so
all numbers above reflect the fixed metric without paying to regenerate
anything. Every flag in this document has been read by hand.
