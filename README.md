# Pretraining Data Optimizer & Micro-Pretraining Benchmark

A lightweight text-cleaning, exact/near-duplicate deduplication pipeline and equal-step character-level micro-pretraining benchmark (`TinyNextTokenNet`) built and tested in Google Colab.

## Pipeline Stages (`pipeline_clean.py`)
1. **Empty & Length Filtering:** Drops blank lines and lines shorter than `MIN_CHARS` (`15`).
2. **Boilerplate / Spam Filtering:** Removes lines matching target noise patterns (e.g., `"click here"`).
3. **Exact Deduplication:** Normalizes whitespace and lowercased word sets, hashing canonical representations with **SHA-256** to drop exact duplicates in constant lookup time.
4. **Near-Duplicate Filtering:** Computes **Jaccard similarity** (`|A ∩ B| / |A ∪ B|`) against kept lines and drops entries meeting or exceeding `SIMILARITY_THRESHOLD`.
5. **Audit Logging:** Records every removed line and filter reason to `dropped_lines.txt`.

## Stress Test: 1 MB Corpus (`9,806` Lines / `151,339` Words)

At `SIMILARITY_THRESHOLD = 0.3`:
- **Surviving output:** `9` lines (`161` words) — **99.9% reduction** (`1,113` characters total).
- **Breakdown of `9,797` dropped lines:** `501` empty, `979` short, `785` spam, `1,446` exact duplicates, and `6,086` near-duplicates.

## Micro-Pretraining Results (`400` Steps, `TinyNextTokenNet`)

| Run Configuration | Total Chars | Train Chars | Val Chars | Start Val Loss | Final Val Loss | vs. Raw |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **RAW (Baseline)** | `1,048,576` | `891,289` | `157,287` | `3.9518` | `1.4258` | — |
| **FILTERED (`threshold = 0.3`)** | `1,113` | `946` | `167` | `3.9581` | `6.6916` | `+5.2658` (worse) |
| **FILTERED (`threshold = 0.7`)** | `6,717` | `5,709` | `1,008` | `3.9543` | `1.1819` | `-0.2439` (better) |

### Key Takeaways
- **Over-pruning starves small datasets:** A `0.3` Jaccard cutoff on a `1 MB` repetitive file left only `1,113` characters (`167` validation characters), causing high validation loss (`6.6916`) due to tiny sample size and split mismatch.
- **Balanced deduplication (`0.7`) improves signal:** Keeping lines below `70%` word overlap retained `6,717` characters (`1,008` validation characters) while stripping spam and heavy repetition, lowering validation loss by **`0.2439`** relative to the raw corpus (`1.1819` vs. `1.4258`).

## Reproduce the Runs
```bash
pip install -r requirements.txt

# 1. Clean raw text (adjust --threshold to 0.3 or 0.7)
python pipeline_clean.py --input stress_1mb.txt --output cleaned_output.txt --threshold 0.7

# 2. Run equal-step micro-pretraining comparison
python micro_pretrain_bench.py --raw stress_1mb.txt --filtered cleaned_output.txt --steps 400
