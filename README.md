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

The dataset shows that documented metadata is missing; five controlled experiments
(Appendix G of the paper) then measure *how much* each undocumented choice actually
moves a score. All runs use `lm-evaluation-harness` + vLLM unless noted.

### 1. Factorial Grid (the framing experiment)

**Setup.** 4 instruction-tuned models (Qwen2.5-14B-Instruct, Qwen2.5-7B-Instruct,
Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.3) × 3 benchmarks (BBH, GSM8K, MMLU)
× 3 prompt formats (`plain` few-shot, `instruct` system-prompt, `cot`
chain-of-thought) × 2 temperatures (0.0, 0.7) × 2 n-shot counts (0 + 3-shot for
BBH; 0 + 5-shot for GSM8K/MMLU) × 3 seeds (42, 123, 7). MMLU reduced to temp 0.0 /
single seed for server constraints. **312 successful runs** (144 BBH + 144 GSM8K +
24 MMLU); GSM8K/MMLU use a 200-problem subset, BBH the full suite. (MMLU cells were
re-scored on the `hf` backend to recover 6 runs that failed under vLLM; log-likelihood
scoring is backend-independent, agreeing within 0.5 pp on cells run both ways.)

**Result — prompt format (one-way ANOVA per benchmark, 5-shot cells).**

| Benchmark | `plain` | `instruct` | `cot` | format effect |
|---|---|---|---|---|
| GSM8K (n=24/cell) | 38.75 | 34.19 | 19.25 | **η²≈27.4%**, F=13.0, p=1.6×10⁻⁵ |
| MMLU (n=4/cell) | 65.67 | 65.87 | 51.95 | η²≈35.1%, F=2.43, p=0.14 (underpowered) |

Prompt format alone explains **27.4%** of GSM8K score variance (plain→cot gap
19.50 pp); temperature explains only η²≈0.25% (F=0.18, p=0.68). MMLU's point estimate
(η²≈35.1%) is even larger but remains underpowered (p=0.14) — the same *ordering* as
GSM8K, the direction robust but the significance not.

**Result — n-shot (ANOVA, pooled by shot count).** GSM8K η²≈**65.1%** (F=264.6,
p<10⁻³⁰) but this is *not* a capability gain: every 0-shot cell scores exactly
**0.00** because instruction-tuned models omit the `####` answer delimiter the
GSM8K parser requires (a parser–format coupling, not learning). MMLU η²≈**0.1%**
(F=0.03, p=0.87; 0-shot 60.17 vs 5-shot 61.16) — on a log-likelihood benchmark
n-shot is nearly inert.

### 2. E1 — GSM8K prompt-format sensitivity

**Setup.** Qwen2.5-14B-Instruct × GSM8K, 5-shot, temp 0.0, seeds 42/123/7
(deterministic, so identical across seeds), `limit=200`.

| Prompt format | Score |
|---|---|
| `plain` (direct-answer few-shot) | **58.5%** |
| `instruct` (system-prompt) | **57.0%** |
| `cot` (chain-of-thought few-shot) | **2.5%** |

**Result.** A single prompt-format change collapses a strong score to near zero:
the CoT free-text output contains no `####` delimiter, so the standard extractor
returns 0 for nearly every item. The same pattern holds at temp 0.7 (58.67 /
52.67 / 2.83).

### 3. E2 — MMLU scoring-mode control

**Setup.** 4 models on MMLU, 5-shot, `plain`, temp 0.0, scored both ways —
log-likelihood vs generation, both on the **full** 14,042-item test set (an exact
same-sample comparison for every model).

| Model | log-likelihood | generation | \|Δ\| |
|---|---|---|---|
| Mistral-7B-Instruct-v0.3 | 61.85 | 61.64 | 0.21 |
| Qwen2.5-7B-Instruct | 74.25 | 74.43 | 0.18 |
| Llama-3.1-8B-Instruct | 68.19 | 68.16 | 0.03 |
| Qwen2.5-14B-Instruct | 79.97 | 79.79 | 0.18 |

**Result.** Max gap **0.21 pp** — scoring mode is nearly inert *on MMLU*. This
rules out a generic scoring-mode effect and shows the ≈35-point BBH gap (case
study §5.1) is specific to lighteval log-likelihood × BBH's CoT structure, not a
property of log-likelihood scoring in general.

### 4. E3 — harness-version sensitivity

**Setup.** Qwen2.5-1.5B-Instruct and Qwen2.5-7B-Instruct on the **full** GSM8K test
set, 5-shot, `hf` backend, `lm-eval` **0.4.3 vs 0.4.11**, seeds 42/123, everything
else held fixed.

| Model | 0.4.3 (s42/s123) | 0.4.11 (s42/s123) |
|---|---|---|
| Qwen2.5-1.5B-Instruct | 32.83 / 33.74 | 33.06 / 33.36 |
| Qwen2.5-7B-Instruct | 75.06 / 76.35 | 75.51 / 78.32 |

**Result.** Largest seed-matched cross-version difference is **1.97 pp** — *below*
the within-version seed spread (up to 2.81 pp). The effect is real but small; more
importantly the two versions are **dependency-irreconcilable** (different
`transformers` stacks), so even "same harness, different version" breaks
comparability — which is why `harness_version` is an EvalSpec Required field.

### 5. Judge-model sensitivity

**Setup.** 5 contestant models each generate one response to the 80 MT-Bench turn-1
questions; the **identical** responses are then scored by 4 judges (GPT-4o-mini,
Gemini-3.5-Flash, Llama-3.1-70B-Instruct, Claude-Haiku-4.5) with the MT-Bench 1–10
rubric at temp 0. Any variation is attributable to the judge alone.

| Contestant | GPT-4o-mini | Gemini-3.5 | Llama-70B | Claude-Haiku | spread |
|---|---|---|---|---|---|
| Gemma-3-4B | 8.24 | 7.96 | 8.31 | 7.26 | 1.04 |
| Qwen2.5-7B | 7.86 | 8.02 | 7.96 | 6.97 | 1.05 |
| Llama-3.1-8B | 8.04 | 7.61 | 8.22 | 6.87 | **1.35** |
| Ministral-8B | 8.44 | 7.54 | 8.62 | 7.51 | 1.11 |
| Qwen2.5-72B | 8.32 | 8.59 | 8.38 | 7.74 | 0.84 |

**Result.** The judge changes the ranking, not just the offset: Ministral-8B is
**1st** under GPT-4o-mini and Llama-70B but **last** under Gemini. Judge-induced
spread averages 1.08 (max 1.35) points, and cross-judge Spearman rank correlation
spans **+1.00 to −0.30** — an LLM-judge score is uninterpretable without recording
the judge.

**Robustness — 8 judge families × 8 tasks** (`experiments/judge_sensitivity_extended/`).
To confirm this is not a 4-judge or single-task artefact, the study was extended to
**8 judge families** (adding DeepSeek, Mistral, Qwen, Amazon) and reported per
MT-Bench task category. The effect *strengthens*: spread rises to **mean 1.66 /
max 1.80**, cross-judge Spearman over the 8 judges spans **+1.00 to −0.10**
(mean +0.61, 28 pairs), and a multi-point spread holds in **all 8 task categories**
(1.5–2.9, largest on coding, extraction, and math).

### 6. GSM8K full-set robustness check (limit=200 vs full)

The factorial grid scored GSM8K on a 200-problem subset (`limit=200`) for compute
budget. To confirm the subset is representative, the Qwen2.5-7B plain-vs-cot cell was
rerun at **full size** (1319 questions) on a Colab A100
(`experiments/controlled_eval/full_vs_limit200_qwen7b_gsm8k.py`):

| format | limit=200 | full (1319) | Δ |
|---|---|---|---|
| plain | 17.00 | 17.29 | **+0.29** |
| cot | 59.50 | 58.23 | **−1.27** |

**Result.** Full-vs-subset deltas are ≤1.3 pp for both formats — **the 200-problem
subset is representative** of the full test set. (This rerun used lm-eval's hf
backend, since vLLM's current wheel targets CUDA-13 vs Colab's CUDA-12; the hf
absolute scores differ from the grid's vLLM scores — itself a backend-as-config
effect — so it validates subset-representativeness rather than reproducing the
vLLM numbers.)

### 7. Prompt-format effect across five log-likelihood benchmarks (beyond GSM8K)

**Setup.** To test whether prompt format matters beyond generation-scored GSM8K, we
measure it on five log-likelihood benchmarks, powering each with the replication unit
its protocol admits: **seeds** (n=12) for the random-few-shot ARC-Challenge,
WinoGrande, and OpenBookQA; **models** (8 instruct models, 1.5B–14B) for the
seed-deterministic MMLU and GPQA. All on a single `hf` backend. Because prompt format
is a *within-subject* factor (each model is scored under plain/instruct/cot), the
test is a **repeated-measures ANOVA** blocking on the subject, cross-checked with the
non-parametric Friedman test.

| Benchmark | powered via | n | RM-ANOVA *p* | verdict |
|---|---|---|---|---|
| ARC-Challenge | seeds | 12 | <1e-4 | significant |
| WinoGrande | seeds | 12 | <1e-4 | significant |
| MMLU | models | 7 | 0.003 | significant |
| OpenBookQA | seeds | 12 | 0.010 | significant |
| GPQA | models | 7 | 0.18 | null (chance floor) |

**Result.** Adequately powered, prompt format has a real effect on log-likelihood
scoring too — **significant on 4 of 5** benchmarks (Friedman-confirmed), null only on
GPQA where the models sit at its ~25% chance floor. Absolute magnitude stays a few
points, an *order below* generation-scored GSM8K's tens — so format matters most where
scoring parses generated text. Data QA: Phi-3.5 excluded (metric did not parse); the
Llama-3.1-8B mirror is flagged but conclusions are robust to excluding it.
(`experiments/controlled_eval/scripts/rm_anova_loglik.py`)

### 8. Additional robustness studies

- **Judge effect, second dataset (Vicuna-Bench)** — the judge-model effect replicates
  on a second dataset: judge-induced spread **1.84**, Friedman χ²(7)=**908.5**
  (Kendall's *W*=0.33), cross-judge Spearman down to **−0.4**.
  (`experiments/judge_sensitivity_vicuna/`)
- **Temperature panel (OpenRouter)** — temperature explains **2.2%** of GSM8K and
  **0.66%** of MT-Bench judged-quality variance, so the "under 1%" finding holds even
  for open-ended, judge-scored generation. (`experiments/temperature_panel/`)
- **EvalSpec emission hook** — a tested drop-in adapter that turns an existing
  lm-evaluation-harness run into EvalSpec-R records, auto-populating **all 8/8
  Required fields** (including `scoring_mode` and the prompt-template hash).
  (`src/emission_hooks/lm_eval_to_evalspec.py`)

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


