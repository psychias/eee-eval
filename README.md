# Every Eval Ever (EEE)

**A unified schema and pipeline for collecting, standardising, and analysing LLM evaluation results at scale.**

Submission for the [EvalEval @ ACL 2026 Shared Task](https://evalevalai.com/events/shared-task-every-eval-ever/) — Track 1 (Public Evaluation Data Parsing).

Companion paper: *"Why LLM Leaderboards Are Incomparable: Structural Fragmentation and Missing Evaluation Metadata at Scale"*

---

## Overview

EEE ingests LLM evaluation results from **11 heterogeneous sources** — 8 leaderboards, HuggingFace model cards, and the Papers With Code archive — into the standardised EEE JSON schema (v0.2.1). The resulting dataset contains **29,331 records** covering **5,672 models** and **180 benchmarks**.

The pipeline then runs cross-source collision detection and metadata coverage analysis to quantify:

1. **Structural fragmentation** — sources evaluate almost entirely non-overlapping model–benchmark combinations (only 16 collision pairs from 29k records)
2. **Missing metadata** — temperature and prompt template are documented in <0.1% of records
3. **Attribution failure** — every cross-source score discrepancy lacks the metadata needed to explain it

## Repository Structure

```
eval.schema.json              # EEE JSON Schema v0.2.1
eval_types.py                 # Pydantic models (auto-generated from schema)
instance_level_types.py       # Instance-level schema types

scripts/
├── extraction/               # Paper extraction engine (Docling + LLM)
│   ├── pipeline.py           #   Orchestration & CLI entry point
│   ├── docling_parser.py     #   PDF table extraction via Docling
│   ├── table_parser.py       #   Results table identification & parsing
│   ├── converter.py          #   Raw data → EEE schema records
│   ├── llm_fallback.py       #   LLM-augmented extraction (OpenRouter)
│   ├── protocol.py           #   Eval protocol extraction (shots, CoT, etc.)
│   ├── normalize.py          #   Benchmark & model name normalisation
│   ├── constants.py          #   Keyword sets, canonical maps
│   ├── helpers.py            #   Cell parsing, score validation
│   ├── download.py           #   arXiv PDF downloader
│   └── prose.py              #   Regex-based prose/caption extraction
├── analysis/                 # Statistical analysis modules
│   ├── collision_detection.py
│   ├── coverage_audit.py
│   ├── rank_instability.py
│   ├── variance_decomposition.py
│   └── ...
├── figures/                  # Publication figure generators
│   ├── fig1_score_deltas.py
│   ├── fig4_coverage_bar.py
│   └── ...
├── scrapers/                 # Leaderboard scrapers
│   ├── hfopenllm_v2_scraper.py
│   ├── alpacaeval2_scraper.py
│   ├── chatbot_arena_scraper.py
│   ├── bigcodebench_scraper.py
│   ├── wildbench_scraper.py
│   ├── mtbench_scraper.py
│   └── base.py
├── add_leaderboard_records.py  # Fetch live leaderboard data → EEE JSON
├── hf_model_card_fetcher.py    # Extract results from HF model cards
├── pwc_fetcher.py              # Fetch Papers With Code results
├── extract_paper.py            # arXiv paper extraction (main entry point)
├── validate_outputs.py         # Schema validation
├── validate_all.py             # Batch validation with auto-fix
├── aggregate_results.py        # Flatten JSON → CSV
├── generate_figures.py         # Generate paper figures
├── fix_data_quality.py         # Post-hoc data quality fixes
├── quality_audit.py            # Comprehensive data quality audit
├── run_hf_pipeline.py          # End-to-end HF data collection
├── convert_eval_logs.py        # Convert eval framework logs → EEE
└── preflight_check.py          # Environment verification

eval_converters/              # Framework log → EEE schema converters
├── lm_eval/                  #   lm-evaluation-harness adapter
├── inspect/                  #   Inspect AI adapter
├── helm/                     #   CRFM HELM adapter
└── common/                   #   Shared base classes & utilities

eee_inspect_adapter/          # Inspect AI scorer → EEE (pip-installable)

utils/                        # Shared utilities & additional adapters
├── helpers/                  #   Schema validation, developer mapping, I/O
├── global-mmlu-lite/         #   Global-MMLU-Lite adapter
├── helm/                     #   HELM leaderboard parser
├── hfopenllm_v2/             #   HF Open LLM v2 adapter
├── proprietary_template/     #   Template for proprietary model evals
├── rewardbench/              #   RewardBench adapter
└── livecodebenchpro/         #   LiveCodeBench-Pro adapter

data/                         # Evaluation records (EEE JSON, per-source)
├── open_llm_leaderboard_v2/  #   26,790 records — lm_eval harness
├── papers_with_code/         #   576 records — mixed harnesses
├── alpacaeval2/              #   446 records — alpaca_eval
├── bfcl/                     #   327 records — gorilla_eval
├── bigcodebench/             #   280 records — bigcodebench
├── chatbot_arena/            #   218 records — fastchat
├── evalplus/                 #   214 records — evalplus
├── hf_model_card/            #   207 records — varies
├── wildbench/                #   122 records — wildeval
├── swe_bench/                #   117 records — swebench
├── mt_bench/                 #   34 records — fastchat
└── aggregated/               #   all_results.csv + coverage_stats.json

figures/                      # Generated analysis figures (PDF)
submission/latex/             # ACL 2026 paper source (LaTeX)
tests/                        # Unit tests for converters and extraction
```

## Quick Start

### Installation

```bash
python -m venv .venv
source .venv/bin/activate         # Linux/macOS
# .venv\Scripts\Activate.ps1      # Windows
pip install -r requirements.txt
```

### Reproduce the Dataset

```bash
# 1. Fetch leaderboard data (requires HF_TOKEN in .env)
python scripts/add_leaderboard_records.py

# 2. Fetch HuggingFace model card results
python scripts/hf_model_card_fetcher.py

# 3. Fetch Papers With Code results
python scripts/pwc_fetcher.py

# 4. Validate all records against the EEE schema
python scripts/validate_outputs.py

# 5. Aggregate into a single CSV
python scripts/aggregate_results.py

# 6. Generate publication figures
python scripts/generate_figures.py
```

### Extract Results from arXiv Papers

```bash
# Requires OPENROUTER_API_KEY in .env for LLM-augmented extraction
python scripts/extract_paper.py --batch scripts/arxiv_ids_full.txt \
    --llm-fallback --llm-model meta-llama/llama-3.3-70b-instruct
```

### Convert Evaluation Framework Logs

```bash
# lm-evaluation-harness
python -m eval_converters.lm_eval --log_path <results.json>

# Inspect AI
python -m eval_converters.inspect --log_path <eval.log>

# HELM
python -m eval_converters.helm --log_path <run_dir/>
```

Each converter accepts `--output_dir`, `--evaluator_relationship`, and other metadata flags. Run with `--help` for details.

## EEE Schema (v0.2.1)

Each evaluation record captures:

| Block | Key Fields |
|---|---|
| **model_info** | name, developer, parameter count, release date |
| **eval_library** | harness name, version |
| **source_metadata** | source URL, source type, evaluator relationship |
| **evaluation_results** | benchmark name, score, metric, generation config (shots, temperature, prompt template, chain-of-thought) |

Schema definition: [`eval.schema.json`](eval.schema.json)

## Key Findings

| Metric | Value |
|---|---|
| Total records | 29,331 |
| Unique models | 5,672 |
| Unique benchmarks | 180 |
| Data sources | 11 |
| Cross-source collision pairs | 16 (8 independent) |
| Temperature coverage | <0.1% |
| Prompt template coverage | <0.1% |

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