# Every Eval Ever (EEE)

**A unified schema and pipeline for collecting, standardising, and analysing LLM evaluation results at scale.**

---

## Overview

EEE ingests LLM evaluation results from heterogeneous sources — a set of public
leaderboards (including HuggingFace model cards and Papers With Code) and an
arXiv-paper LLM-extraction source — into the standardised EEE JSON schema
(v0.2.1). The aggregated dataset spans **12 sources** and contains **46,559
records** covering **7,398 unique models** and **1,954 unique benchmarks**.

The pipeline then runs cross-source collision detection and metadata-coverage
analysis to quantify:

1. **Structural fragmentation** — sources evaluate almost entirely
   non-overlapping model–benchmark combinations.
2. **Missing metadata** — temperature and prompt template are documented in a
   vanishingly small fraction of records.
3. **Attribution failure** — cross-source score discrepancies lack the metadata
   needed to explain them.

## What the paper shows

**Companion paper:** *"Why LLM Leaderboards Are Incomparable: Structural
Fragmentation and Missing Evaluation Metadata at Scale."*

**The core idea.** A single number like "MMLU = 68" is only comparable across
models if it was produced the same way. In practice it almost never is: the same
model × benchmark is scored with different harnesses, n-shot counts, prompt
formats, temperatures, and scoring modes (log-likelihood vs. generation), and
those choices move scores by more than the model-to-model gaps a leaderboard
claims to measure. EEE assembles 46,559 evaluation records from 12 sources into
one schema to make this measurable, and shows that leaderboards are
*incomparable* for two structural reasons — and that the information needed to
fix it is systematically missing.

**The dataset.**

| | Records | Models | Benchmarks |
|---|---|---|---|
| **Full dataset (12 sources)** | **46,559** | **7,398** | **1,954** |
| Leaderboard subset (10 sources) | 28,755 | 5,301 | 174 |
| ArXiv extraction (100 papers) | 17,228 | 1,726 | 1,806 |
| Papers With Code | 576 | 383 | 16 |

**Finding 1 — structural fragmentation.** Sources evaluate almost entirely
non-overlapping model × benchmark cells; the leaderboard ecosystem (dominated by
Open LLM Leaderboard v2, 26,790 records) is largely an island with near-zero
model overlap with the other sources. Most "leaderboards" cannot be cross-checked
because they never measure the same things.

**Finding 2 — missing metadata (the headline).** The configuration fields most
responsible for score divergence are the least reported, and are *entirely absent*
from leaderboards:

| Field | Full dataset | Leaderboard subset |
|---|---|---|
| n-shot | 80.5% | 98.0% |
| harness | 63.2% | 93.2% † |
| chain-of-thought | 50.0% | 77.7% |
| prompt template | 14.1% | **0.0%** |
| temperature | 3.1% | **0.0%** |

† Leaderboard harness coverage is pipeline-attached from platform documentation
(e.g. `lighteval`), not exposed by any leaderboard API. `prompt_template` and
`temperature` have **zero** coverage across the entire leaderboard subset.

**Finding 3 — divergence is unattributable.** When the same model × benchmark *is*
reported by multiple sources, the scores frequently disagree (case studies in §5:
BBH, GSM8K, TruthfulQA, Belebele, HellaSwag), and because the explanatory metadata
is missing, the gap cannot be attributed to a cause. Several "independent"
agreements turn out to be the *same* pipeline reported twice (e.g. OLv2 and
SEA-LION matching to two decimals on Gemma-2-9B BBH).

**Extraction audit (validating the arXiv pipeline against the source papers,
10 papers, human-verified).**

- **Score extraction:** **97.5%** precision (1,977/2,027 extracted scores match
  the paper) and **≈81%** recall (the pipeline captures headline tables but skips
  many secondary ones).
- **Harness identification:** **87.1%** per-row precision on the with-harness pool
  (817 rows); **100%** correct abstention on the without-harness pool — the
  pipeline never hallucinates a harness.

**EvalSpec v0.1** (the proposed remedy): a reporting specification of 9 Required +
6 Recommended + 3 Full fields, so that the fields driving divergence become a
recordable absence rather than an invisible gap.

## Experiment outcomes

Controlled experiments isolate *how much* each undocumented choice moves a score
(full setup in Appendix G of the paper):

- **Factorial grid** (4 models × 3 benchmarks × 3 prompt formats × 2 temperatures
  × 2 n-shots × 3 seeds; 306 successful runs). Prompt format alone explains
  **η² = 27.4%** of GSM8K score variance (F = 13.0, p < 0.001); n-shot explains
  **η² = 65.1%**. On MMLU the format effect is descriptively consistent but
  underpowered (η² = 24.5%, not significant) — the magnitude is benchmark-specific,
  the direction is not.
- **E1 — GSM8K prompt format** (Qwen2.5-14B, 5-shot): plain **58.5**, instruct
  **57.0**, cot **2.5**. A format/answer-parser interaction can collapse an
  otherwise-strong score to near zero.
- **E2 — MMLU scoring mode** (log-likelihood vs. generation, 4 models): max gap
  **0.28 pp** — for MMLU specifically, scoring mode barely matters, showing the
  effects are benchmark-dependent rather than universal.
- **E3 — harness version** (lm-eval 0.4.3 vs. 0.4.11, GSM8K): seed-matched
  cross-version differences are small (< 0.5 pp), but the two versions are
  **dependency-irreconcilable** (different transformers stacks), so even "same
  harness" is not a guarantee of comparability.
- **Judge-model sensitivity** (MT-Bench, identical responses scored by 4 judges):
  changing *only the judge* reorders the leaderboard — cross-judge Spearman rank
  correlation ranges from **+1.00 to −0.30** (mean spread 1.08 points).

## Dataset
The aggregated dataset is available on HuggingFace:
- Dataset: https://huggingface.co/datasets/evaleval/EEE_datastore

## Repository layout

```
eval.schema.json                       EEE JSON Schema v0.2.1
src/eee_eval/eval_types.py             Pydantic models for the schema
src/eee_eval/instance_level_types.py   Instance-level schema types

scripts/reproduce_paper.py        Recompute every paper number → analysis_output/paper_numbers.json
scripts/reproduce_case_studies.py Re-derive the §5 case-study numbers from data/
scripts/generate_evidence_file.py Build section5_evidence.txt (raw-data provenance)

src/eee_eval/
├── extraction/               Ingestion: arXiv LLM extraction + leaderboard/model-card fetchers
├── analysis/                 aggregate_results (→ all_results.csv) + coverage_audit
├── converters/               Framework-log → EEE adapters (lm_eval, inspect, helm, common)
├── scrapers/                 Leaderboard-specific scrapers
├── validation/               Schema validation + output checks
└── utils/                    Shared helpers (I/O, schema validation)

data/                         Aggregated EEE records, per source (gitignored)
experiments/                  Controlled experiments + notebooks
tests/                        Unit + integration tests
analysis_output/              Generated stats, incl. paper_numbers.json
```

## Installation

Python 3.10 or newer (the code is verified on 3.10).

```bash
python -m venv .venv
source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\Activate.ps1       # Windows PowerShell
pip install -r requirements.txt
```

## Environment variables

Copy `.env.example` to `.env` and fill in the keys you need:

| Variable | Required for | Notes |
|---|---|---|
| `OPENROUTER_API_KEY` | `eee_eval.extraction.extract_paper` | LLM extraction of results from arXiv papers |
| `HF_TOKEN` | `eee_eval.extraction.add_leaderboard_records`, `hf_model_card_fetcher` | HuggingFace access token for the leaderboard / model-card fetchers |
| `REQUESTS_CA_BUNDLE` | optional | Custom CA bundle if TLS verification fails in your environment |

Reproducing the paper's numbers does **not** require any keys — the raw data
ships in `data/`.

## Reproducing the paper's numbers

`reproduce_paper.py` is the single source of truth. It recomputes every
quantitative claim from the raw data files and writes them to
`analysis_output/paper_numbers.json`:

```bash
python scripts/reproduce_paper.py              # recompute and write paper_numbers.json
python scripts/reproduce_paper.py --no-rebuild # reuse the existing data/aggregated/all_results.csv
# or: make repro
```

It covers §3.2 dataset totals and per-source counts, §4.1 score-extraction
audit, §4.2 harness audit, §4.3 metadata coverage, §5 factorial-grid ANOVA, and
the appendix experiments. For richer raw-data provenance (the underlying JSON
paths and harness/shot fields behind each case study), run:

```bash
python scripts/reproduce_case_studies.py       # §5 case-study numbers
python scripts/generate_evidence_file.py       # writes section5_evidence.txt
```

## Rebuilding the dataset (optional)

The aggregated `data/` directory is already provided. To re-fetch from source:

```bash
# Leaderboard records (needs HF_TOKEN)
python -m eee_eval.extraction.add_leaderboard_records

# HuggingFace model-card results (needs HF_TOKEN)
python -m eee_eval.extraction.hf_model_card_fetcher

# Papers With Code results
python -m eee_eval.extraction.pwc_fetcher

# Extract results from arXiv papers (needs OPENROUTER_API_KEY)
python -m eee_eval.extraction.extract_paper --paper-list papers/general_llm_papers.txt

# Validate records against the EEE schema
python -m eee_eval.validation.validate_outputs

# Aggregate into a single CSV (data/aggregated/all_results.csv)
python -m eee_eval.analysis.aggregate_results
```

## Analysis and figures

```bash
# Metadata coverage audit (→ analysis_output/coverage_stats.csv)
python -m eee_eval.analysis.coverage_audit

# Regenerate the paper figures
python LLM_Evaluation_Report/figures/scripts/coverage_bars.py   # Figure 6 (§4.3)
python LLM_Evaluation_Report/figures/scripts/unified.py         # Figure 5 (§5)
python LLM_Evaluation_Report/figures/scripts/case1.py           # Figure 4 (§5.1)
```

## Converting evaluation-framework logs

Each converter has its own CLI. Run any of them with `--help` for the full flag
list (`--output_dir`, `--evaluator_relationship`, etc.):

```bash
# lm-evaluation-harness
python -m eee_eval.converters.lm_eval --log_path <results.json>

# Inspect AI
python -m eee_eval.converters.inspect --log_path <eval.log>

# CRFM HELM
python -m eee_eval.converters.helm --log_path <run_dir/>
```

## Running the tests

```bash
pip install pytest
pytest
```

## EEE Schema (v0.2.1)

Each evaluation record captures:

| Block | Key fields |
|---|---|
| **model_info** | name, developer, parameter count, release date |
| **eval_library** | harness name, version |
| **source_metadata** | source URL, source type, evaluator relationship |
| **evaluation_results** | benchmark name, score, metric, generation config (shots, temperature, prompt template, chain-of-thought) |

Schema definition: [`eval.schema.json`](eval.schema.json) · Pydantic models:
[`src/eee_eval/eval_types.py`](src/eee_eval/eval_types.py).

## Key findings

| Metric | Value |
|---|---|
| Total records | 46,559 |
| Unique models | 7,398 |
| Unique benchmarks | 1,954 |
| Data sources | 12 |


