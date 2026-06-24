# lm-evaluation-harness — version comparison (0.4.3 vs 0.4.11)

Finishes the unfinished `negative_control_version` experiment: instead of comparing
two *harnesses*, this compares two *versions of the same harness*
(EleutherAI `lm-evaluation-harness`) under identical config, so any score delta is
attributable to the version bump alone.

Design is borrowed from the TRL-on-Colab-CLI flow
(<https://huggingface.co/blog/sergiopaniego/trl-colab-cli>): each job is one
`colab run` against a **fresh ephemeral VM**. The two lm-eval versions can't share a
Python env, so each (version, model) pair installs its pinned version on its own VM.

## Matrix
- **Versions:** `0.4.3` vs `0.4.11`   (backend `--model hf`)
- **Models:** `Qwen/Qwen2.5-1.5B-Instruct` (small) + `Qwen/Qwen2.5-7B-Instruct` (mid)
- **Task:** `gsm8k`, `num_fewshot=5`, `limit=200`, `seed=42`
- 2 versions × 2 models = **4 ephemeral jobs**

## Files
| File | Runs on | Role |
|------|---------|------|
| `run_lm_eval_version.py` | remote Colab VM | installs one pinned lm-eval, runs one eval, writes a JSONL record |
| `orchestrate.sh`         | your Linux/macOS/WSL shell | loops version × model, calls `colab run`, combines outputs |
| `compare_versions.py`    | locally (Windows OK) | pairs A/B, writes `version_comparison_report.txt` + `version_pairs.csv` |

## Prereqs (one-time)
> ⚠️ `google-colab-cli` is **Linux/macOS only** — run the `colab …` steps from WSL,
> a Mac, or a Linux box, not Windows PowerShell. `compare_versions.py` runs anywhere.

```bash
uv tool install google-colab-cli
colab auth --auth oauth2          # interactive Google login (uses your credits)
```

## Run
```bash
# from WSL/Linux/macOS, inside version_compare/
bash orchestrate.sh               # GPU defaults to A100 (you have credits)
# override knobs if needed:
GPU=L4 LIMIT=0 NUM_FEWSHOT=8 bash orchestrate.sh
```
Each job: fresh VM → `pip install lm-eval==<ver>` → `lm_eval --model hf ...` →
record retrieved to `results/<model>_gsm8k_<ver>.jsonl`. `orchestrate.sh` then
concatenates them into `results/all_version_runs.jsonl`.

## Analyze
```bash
python compare_versions.py results/all_version_runs.jsonl
# -> version_comparison_report.txt  (delta per model, overall mean/median |Δ|)
# -> version_pairs.csv
```

## Notes
- Failed jobs (OOM, install error) are still written as `status:error` rows — they're
  reported as "failed runs" and excluded from pairing, so one bad job won't poison the diff.
- To widen the version gap, set `VERSIONS=("0.4.3" "0.4.11")` in `orchestrate.sh` to the
  two pip versions / git tags you want. The runner takes any `--lm-eval-version` pip spec.
- This is the same record schema as `../negative_control_version.jsonl`, so downstream
  analysis in `eee_eval.ipynb` can ingest it the same way.
