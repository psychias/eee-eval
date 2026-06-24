# Repository layout

```
src/eee_eval/          importable package (src layout)
  extraction/          ArXiv + leaderboard + HF model-card ingestion
  converters/          framework-log -> EEE schema adapters (lm_eval/helm/inspect)
  scrapers/            per-leaderboard scrapers
  analysis/            produces the paper's numbers (coverage, collisions, stats)
  validation/          schema validation + quality audits
  utils/               shared helpers
  eval_types.py        pydantic EEE schema models
scripts/               thin CLI entry points (reproduce_paper.py, ...)
configs/               hyperparameters + per-experiment run logs (not in code)
tests/                 mirrors src/ (pytest)
experiments/           controlled experiments (factorial grid, E1-E4, judge)
notebooks/             exploration only (never imported by the package)
data/ outputs/ analysis_output/   gitignored, regenerated
docs/                  this file + refactoring_plan.md
```

Reproduce every paper number: `python scripts/reproduce_paper.py`
(writes `analysis_output/paper_numbers.json`, the single source of truth).
