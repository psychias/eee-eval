# Every Eval Ever (EEE)

**A unified schema and pipeline for collecting, standardising, and analysing LLM evaluation results at scale.**

Submission for the [EvalEval @ ACL 2026 Shared Task](https://evalevalai.com/events/shared-task-every-eval-ever/) — Track 1 (Public Evaluation Data Parsing).

Companion paper: *"Why LLM Leaderboards Are Incomparable: Structural Fragmentation and Missing Evaluation Metadata at Scale."*

---

## Overview

EEE ingests LLM evaluation results from heterogeneous sources — a set of public
leaderboards (including HuggingFace model cards and Papers With Code) and an
arXiv-paper LLM-extraction source — into the standardised EEE JSON schema
(v0.2.1). The aggregated dataset spans **12 sources** and contains **37,718
records** covering **6,455 unique models** and **1,289 unique benchmarks**.

The pipeline then runs cross-source collision detection and metadata-coverage
analysis to quantify:

1. **Structural fragmentation** — sources evaluate almost entirely
   non-overlapping model–benchmark combinations.
2. **Missing metadata** — temperature and prompt template are documented in a
   vanishingly small fraction of records.
3. **Attribution failure** — cross-source score discrepancies lack the metadata
   needed to explain them.

All of these numbers are recomputed from the raw data by `reproduce_paper.py`
(see [Reproducing the paper's numbers](#reproducing-the-papers-numbers)).

## Repository layout

```
eval.schema.json              EEE JSON Schema v0.2.1
eval_types.py                 Pydantic models for the schema
instance_level_types.py       Instance-level schema types

reproduce_paper.py            Recompute every paper number → analysis_output/paper_numbers.json
reproduce_case_studies.py     Re-derive the §5 case-study numbers from data/
generate_evidence_file.py     Build section5_evidence.txt (raw-data provenance)

src/
├── extraction/               Ingestion: arXiv LLM extraction + leaderboard/model-card fetchers
├── analysis/                 Statistical analysis (coverage, collisions, variance, ranking)
├── converters/               Framework-log → EEE adapters (lm_eval, inspect, helm, common)
├── scrapers/                 Leaderboard-specific scrapers
├── validation/               Schema validation + preflight checks
├── utils/                    Shared helpers (I/O, schema validation)
└── figures/                  Figure generators

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
| `OPENROUTER_API_KEY` | `src/extraction/extract_paper.py` | LLM extraction of results from arXiv papers |
| `HF_TOKEN` | `src/extraction/add_leaderboard_records.py`, `hf_model_card_fetcher.py` | HuggingFace access token for the leaderboard / model-card fetchers |
| `REQUESTS_CA_BUNDLE` | optional | Custom CA bundle if TLS verification fails in your environment |

Reproducing the paper's numbers does **not** require any keys — the raw data
ships in `data/`.

## Reproducing the paper's numbers

`reproduce_paper.py` is the single source of truth. It recomputes every
quantitative claim from the raw data files and writes them to
`analysis_output/paper_numbers.json`:

```bash
python reproduce_paper.py              # recompute and write paper_numbers.json
python reproduce_paper.py --no-rebuild # reuse the existing data/aggregated/all_results.csv
```

It covers §3.2 dataset totals and per-source counts, §4.1 score-extraction
audit, §4.2 harness audit, §4.3 metadata coverage, §5 factorial-grid ANOVA, and
the appendix experiments. For richer raw-data provenance (the underlying JSON
paths and harness/shot fields behind each case study), run:

```bash
python reproduce_case_studies.py       # §5 case-study numbers
python generate_evidence_file.py       # writes section5_evidence.txt
```

## Rebuilding the dataset (optional)

The aggregated `data/` directory is already provided. To re-fetch from source:

```bash
# Leaderboard records (needs HF_TOKEN)
python src/extraction/add_leaderboard_records.py

# HuggingFace model-card results (needs HF_TOKEN)
python src/extraction/hf_model_card_fetcher.py

# Papers With Code results
python src/extraction/pwc_fetcher.py

# Extract results from arXiv papers (needs OPENROUTER_API_KEY)
python src/extraction/extract_paper.py --paper-list papers/general_llm_papers.txt

# Validate records against the EEE schema
python src/validation/validate_outputs.py

# Aggregate into a single CSV
python src/analysis/aggregate_results.py
```

## Analysis and figures

```bash
# Metadata coverage audit
python src/analysis/coverage_audit.py

# Comprehensive cross-source analysis
python src/analysis/run_analysis.py

# Regenerate publication figures
python src/figures/generate_all_figures.py
```

## Converting evaluation-framework logs

Each converter has its own CLI. Run any of them with `--help` for the full flag
list (`--output_dir`, `--evaluator_relationship`, etc.):

```bash
# lm-evaluation-harness
python -m src.converters.lm_eval --log_path <results.json>

# Inspect AI
python -m src.converters.inspect --log_path <eval.log>

# CRFM HELM
python -m src.converters.helm --log_path <run_dir/>
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
[`eval_types.py`](eval_types.py).

## Key findings

| Metric | Value |
|---|---|
| Total records | 37,718 |
| Unique models | 6,455 |
| Unique benchmarks | 1,289 |
| Data sources | 12 |

(Recompute these with `python reproduce_paper.py`.)

## Citation

```bibtex
@inproceedings{eee-2026,
  title     = {Why {LLM} Leaderboards Are Incomparable: Structural Fragmentation
               and Missing Evaluation Metadata at Scale},
  author    = {Anonymous},
  booktitle = {Proceedings of the EvalEval Workshop at ACL 2026},
  year      = {2026},
}
```

## License

See [LICENSE](LICENSE).

## Links

- Dataset: https://huggingface.co/datasets/evaleval/EEE_datastore
- Shared task: https://evalevalai.com/events/shared-task-every-eval-ever/
