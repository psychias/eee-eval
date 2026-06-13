# Every Eval Ever (EEE)

**A unified schema and pipeline for collecting, standardising, and analysing LLM evaluation results at scale.**

Submission for the [EvalEval @ ACL 2026 Shared Task](https://evalevalai.com/events/shared-task-every-eval-ever/) — Track 1 (Public Evaluation Data Parsing).

Companion paper: *"Why LLM Leaderboards Are Incomparable: Structural Fragmentation and Missing Evaluation Metadata at Scale"*

---

## Overview

EEE ingests LLM evaluation results from **11 heterogeneous sources** — 10 leaderboards (incl. HuggingFace model cards) and an ArXiv-paper LLM-extraction source — into the standardised EEE JSON schema (v0.2.1). The resulting dataset contains **43,788 records** covering approximately **7,000 unique models** and **1,400 unique benchmarks**.

The pipeline then runs cross-source collision detection and metadata coverage analysis to quantify:

1. **Structural fragmentation** — sources evaluate almost entirely non-overlapping model–benchmark combinations (only 16 collision pairs from 29k records)
2. **Missing metadata** — temperature and prompt template are documented in <0.1% of records
3. **Attribution failure** — every cross-source score discrepancy lacks the metadata needed to explain it

## Repository Structure

```
eval.schema.json              # EEE JSON Schema v0.2.1
eval_types.py                 # Pydantic models (auto-generated from schema)
instance_level_types.py       # Instance-level schema types

src/                          # All source code
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
│   ├── gen_scatter.py        #   Metadata completeness vs. delta scatter
│   ├── gen_sensitivity.py    #   Cinelli-Hazlett sensitivity contour
│   ├── gen_config_vs_crosssource.py  # Config effect vs. cross-source
│   └── ...
├── scrapers/                 # Leaderboard scrapers
│   ├── hfopenllm_v2_scraper.py
│   ├── alpacaeval2_scraper.py
│   ├── chatbot_arena_scraper.py
│   └── base.py
├── converters/               # Framework log → EEE schema converters
│   ├── lm_eval/              #   lm-evaluation-harness adapter
│   ├── inspect/              #   Inspect AI adapter
│   ├── helm/                 #   CRFM HELM adapter
│   └── common/               #   Shared base classes & utilities
├── inspect_adapter/          # Inspect AI scorer → EEE (pip-installable)
├── utils/                    # Shared utilities & additional adapters
│   ├── helpers/              #   Schema validation, developer mapping, I/O
│   ├── helm/                 #   HELM leaderboard parser
│   ├── hfopenllm_v2/        #   HF Open LLM v2 adapter
│   └── ...
├── extraction/               # Extraction scripts
│   ├── extract_paper.py      #   arXiv paper extraction (main entry point)
│   ├── constants.py          #   Canonical names, developer maps
│   ├── add_leaderboard_records.py  # Leaderboard fetcher
│   ├── hf_model_card_fetcher.py    # Model card extractor
│   ├── pwc_fetcher.py        #   Papers With Code fetcher
│   └── ...
├── validation/               # Validation scripts
│   ├── validate_outputs.py   #   Schema validation
│   ├── quality_audit.py      #   Data quality checks
│   └── preflight_check.py    #   Environment verification
├── analysis/                 # Analysis scripts
│   ├── aggregate_results.py  #   Flatten JSON → CSV
│   ├── run_analysis.py       #   Tasks 1-5 comprehensive analysis
│   ├── collision_detection.py #  Collision-pair analysis
│   └── ...
├── figures/                  # Figure generation
│   └── generate_all_figures.py

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

outputs/                      # Task analysis outputs (tables, reports)
analysis_output/              # Statistical analysis CSV results
harness_experiment_output/    # Harness re-evaluation experiment data
experiments/                  # Controlled evaluation experiment
├── results/                  #   JSONL results + LaTeX macros
└── figures/                  #   Experiment figures
evalspec/                     # Schema validation framework
submission/latex/             # ACL 2026 paper source (LaTeX)
├── figures/                  #   All publication figures (PDF/PNG)
└── experiments/              #   Robustness experiment scripts
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
python src/extraction/add_leaderboard_records.py

# 2. Fetch HuggingFace model card results
python src/extraction/hf_model_card_fetcher.py

# 3. Fetch Papers With Code results
python src/extraction/pwc_fetcher.py

# 4. Validate all records against the EEE schema
python src/validation/validate_outputs.py

# 5. Aggregate into a single CSV
python src/analysis/aggregate_results.py

# 6. Generate publication figures
python src/figures/generate_all_figures.py
```

### Extract Results from arXiv Papers

```bash
# Requires OPENROUTER_API_KEY in .env for LLM-augmented extraction
python src/extraction/extract_paper.py --batch papers/general_llm_papers.txt \
    --llm-fallback --llm-model meta-llama/llama-3.3-70b-instruct
```

### Verify Every Number in the Paper

A single end-to-end script re-derives every quantitative claim in the paper
from the raw data files and reports PASS/FAIL per claim:

```bash
python reproduce_paper.py             # full PASS/FAIL report
python reproduce_paper.py --strict    # exit 1 if any check fails
```

The script verifies §3.2 dataset totals + Table 1 per-source counts, §4.1
score-extraction audit (1,619 entries, Table 7), §4.2 harness audit (per-paper
precision for BLOOM, Llemma, Sheared LLaMA, Aya 23, Nemotron-4 15B), §4.3
metadata coverage (Figure 6 / Table 9), §5 factorial-grid ANOVA stats
(F, p, η² for GSM8K and MMLU), Appendix E2 MMLU scoring-mode gaps, Appendix G
per-format means, and delegates §5 case-study scores to
`reproduce_case_studies.py`. Numbers from external citations (e.g. the
MATH-Verify HF blog post) are documented but not locally recomputable.

For richer raw-data provenance (the underlying JSON paths and harness/shot
fields backing each case study), run `python generate_evidence_file.py` to
write `section5_evidence.txt`.

### Convert Evaluation Framework Logs

```bash
# lm-evaluation-harness
python -m src.converters.lm_eval --log_path <results.json>

# Inspect AI
python -m src.converters.inspect --log_path <eval.log>

# HELM
python -m src.converters.helm --log_path <run_dir/>
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
| Total records | 43,788 |
| Unique models | ~7,000 |
| Unique benchmarks | ~1,400 |
| Data sources | 11 |
| 10-source leaderboard subset records | 28,755 |
| ArXiv-extraction records | 14,457 (100 papers) |
| Score-extraction audit agreement | 100% (1,619 entries / 10 papers) |
| Harness-identification audit precision | 86.2% (817-row WITH sample) |
| Harness-identification abstention quality | 100% (816-row WITHOUT sample) |
| Temperature coverage (10-source) | 0.0% |
| Prompt template coverage (10-source) | 0.0% |

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



https://huggingface.co/blog/math_verify_leaderboard
https://github.com/huggingface/lighteval

https://huggingface.co/datasets/evaleval/EEE_datastore
https://evalevalai.com/about/
https://evalevalai.com/infrastructure/2026/02/17/everyevalever-launch/