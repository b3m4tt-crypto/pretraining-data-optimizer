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
## Near-Duplicate Filtering & Micro-Pretraining Benchmark (`stress_1mb.txt`)

Evaluated on `stress_1mb.txt` (`1,048,576` bytes, `471` parsed documents) using Jaccard similarity near-duplicate filtering and a small character-level transformer (`TinyCharLM`, `block_size = 64`, `batch_size = 32`, `lr = 3e-3`) evaluated on a deduplicated validation split (`71` raw val docs $\rightarrow$ `37` deduplicated val docs; `25,665` validation tokens).

### 1. Jaccard Threshold Sweep (`471` Docs)

| Threshold | Kept Docs | Dropped Docs | % Kept | Notes |
| :--- | :---: | :---: | :---: | :--- |
| `0.3` | 15 | 456 | ~3.2% | Over-aggressive on a `1 MB` stress set; drops most distinct blocks |
| `0.6` | 117 | 354 | ~24.8% | Practical baseline balancing redundancy removal and retention |
| `0.7` | 212 | 259 | ~45.0% | Lighter filtering; allows more near-duplicate phrasing through |

### 2. Equal-Epoch Micro-Pretraining (`5` Epochs on Deduplicated Val)

| Epoch | Raw Val Loss (`917,930` train tokens) | 0.6 No-Clean Val Loss (`74,245` train tokens) | 0.6 Pre-Cleaned Val Loss (`54,994` train tokens) | Clean vs. No-Clean |
| :---: | :---: | :---: | :---: | :---: |
| **1** | 0.3584 | 2.4759 | 2.5478 | +0.0719 |
| **2** | 0.2045 | 2.2785 | 2.3362 | +0.0577 |
| **3** | 0.1850 | 2.1372 | 2.2259 | +0.0888 |
| **4** | 0.1763 | 1.9717 | 2.1378 | +0.1661 |
| **5** | 0.1712 | 1.7367 | 2.0246 | +0.2879 |

### 3. Pre-Cleaning Pass (`0.6` Threshold)

* **Filler & Reference Stripping:** Removing conversational fillers (`ok`, `lol`, `yup`, etc.) and `[Ref #...]` tags before `0.6` Jaccard filtering reduced kept documents from **117** to **99** (`21.0%` kept) and narrowed the training split from **74,245** to **54,994** characters.
* **Key Observations:**
  * Deduplicating the validation split prevents repeated boilerplate from artificially inflating raw validation scores.
  * Every training run steadily lowers validation loss across epochs 1–5 (`0.6 Pre-Cleaned` drops from `2.5478` to `2.0246`, a `-0.5232` reduction).
  * On a small `~1 MB` dataset trained from scratch for 5 epochs, aggressive filtering and pre-cleaning shrink total token volume and gradient steps per epoch compared to unfiltered text—whereas on larger datasets, removing boilerplate and near-duplicates saves compute and storage without wasting updates on redundant blocks.
