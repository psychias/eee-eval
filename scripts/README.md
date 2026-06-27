# `scripts/` — reproduction & verification entry points

Thin, self-contained command-line scripts. Each runs from the repo root, reads
the on-disk `data/` and `experiments/` outputs, and writes to `analysis_output/`
(or repo root). None requires API keys — they recompute from shipped data.

| Script | What it does | Output |
|---|---|---|
| `reproduce_paper.py` | **Single source of truth.** Rebuilds the canonical aggregate from `data/` and recomputes every headline number (§3 dataset totals, §4 coverage + audits, §5/App-G experiments). | `analysis_output/paper_numbers.json` (+ printed report) |
| `reproduce_case_studies.py` | Re-derives the §5 case-study numbers from the raw per-record JSONs and experiment JSONLs — 19 independent checks (`PASS`/`FAIL`). | stdout (19/19 verification) |
| `generate_evidence_file.py` | Builds the raw-data provenance dump behind the §5 case studies (JSON paths, harness/shot fields per claim). | `section5_evidence.txt` (repo root) |

## Usage

```bash
python scripts/reproduce_paper.py              # recompute all paper numbers
python scripts/reproduce_paper.py --no-rebuild # reuse data/aggregated/all_results.csv (faster)
python scripts/reproduce_case_studies.py       # §5 case-study checks (19/19)
python scripts/generate_evidence_file.py       # write section5_evidence.txt
# or: make repro   (== reproduce_paper.py)
```

`reproduce_paper.py` is the authoritative one; the other two are detail/provenance
companions for §5. See the top-level `README.md` for the full reproduction guide.
