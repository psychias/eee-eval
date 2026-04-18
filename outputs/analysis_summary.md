# EEE-Eval Analysis Summary

Generated: 2026-04-15

## Task 1 — Permutation Null Model

A permutation test (n=10,000) replaced the birthday-paradox approximation for expected
cross-source collision counts. With all sources, 6 observed collisions were found
against a permutation mean of 0.9 (SD=0.8), yielding z=6.0
and p=1.9e-09. Excluding OLv2, the observed count drops to 5 against
a null mean of 0.8 (z=5.4, p=5.3e-08).
The observed collision count significantly exceeds
the permutation null in both subsets, confirming that
cross-source model-benchmark overlap is non-random.

## Task 2 — OLv2-Exclusion as Primary Finding

OLv2 contributes 91.3% of all records but its model population is largely isolated
(max Jaccard = 0.010). Removing OLv2 drops harness coverage from
97.6% to 71.8%, revealing weaker metadata norms among non-OLv2 sources.
The Jaccard heatmap and benchmark overlap matrix confirm that OLv2 operates as a parallel
evaluation ecosystem. This warrants elevation from sensitivity analysis to a primary finding.

## Task 3 — Missing Metadata Cost (Incentive Problem)

Most sources achieve zero coverage for temperature and prompt template fields. Several sources
that already log harness name and n-shot (implying per-run infrastructure exists) still omit
temperature, making them high-leverage targets for a 30-minute one-time setup. The documentation
gap is largest for OLv2 (26,790 records) and
PWC (570 records).

## Task 4 — PWC Artefact Analysis

Of 576 PWC records, 2 (0.3%) were flagged as artefacts:
0 with score > 1000 (possible param counts), 2 with variant-tag
leakage, and 0 exact duplicates. A best-practices document with detection rules
and an ingestion checklist was produced.

## Task 5 — Reproducibility Checklist

Eight fields were assessed for reproducibility documentation. Current ecosystem coverage ranges from
0.0% to 97.6%.
The minimum viable report (coverage > 50% or difficulty = Low) includes:
eval_library.name, eval_library.version, generation_args.n_shot, additional_details.cot.

## Output Files

| File | Description |
|------|-------------|
| `task1_permutation_results.csv` | Permutation test statistics (all sources, excl OLv2) |
| `task1_permutation_table.tex` | LaTeX table for permutation null model |
| `task1_permutation_histograms.pdf/png` | Permutation null distribution plots |
| `task2_coverage_by_subset.csv` | Metadata coverage by source-exclusion subset |
| `task2_coverage_table.tex` | LaTeX Table 3 reproduction |
| `task2_jaccard_models.csv` | Jaccard similarity matrix (model sets) |
| `task2_jaccard_models_heatmap.pdf/png` | Jaccard similarity heatmap |
| `task2_benchmark_overlap.csv` | Benchmark overlap matrix |
| `task2_benchmark_overlap_heatmap.pdf/png` | Benchmark overlap heatmap |
| `task2_olv2_isolation_paragraph.txt` | Plain-English paragraph for Section 5.4 |
| `task3_documentation_gap.csv` | Documentation gap by source and field |
| `task3_gap_table.tex` | LaTeX table of documentation gaps |
| `task3_documentation_gap_barchart.pdf/png` | Horizontal bar chart of gaps |
| `task4_pwc_bench_artefact_rate.csv` | Per-benchmark artefact rate in PWC |
| `pwc_best_practices.md` | PWC ingestion best practices document |
| `task5_reproducibility_checklist.csv` | Reproducibility checklist data |
| `task5_checklist_table.tex` | LaTeX longtable for checklist |
| `task5_reproducibility_checklist.md` | Markdown checklist for dataset README |
| `analysis_summary.md` | This file |

## Data Quality Issues

- `temperature`: 100.0% missing/unknown across all sources
- `prompt_template`: 100.0% missing/unknown across all sources

- The `eval_library` column in the flat CSV merges library name and version; version-level
  granularity requires parsing individual JSON files.
- The `scoring_mode` and `provenance_link` fields do not exist in the current schema/CSV;
  coverage is reported as 0%.
- OLv2 records use `shots` = 0/5/25 but leave `temperature`, `prompt_template`, and
  `chain_of_thought` systematically blank.

## Suggested Next Steps

1. **Adopt the permutation null model** in Section 4 to replace the birthday-paradox formula,
   which under-counts expected collisions due to source-specific benchmark scope.
2. **Promote OLv2 isolation** to a primary finding (Section 5.4) with the provided paragraph
   and heatmaps.
3. **Add the documentation gap table** (Task 3) to the recommendations section to concretely
   quantify the effort needed for field-level metadata improvements.
4. **Include the PWC best-practices note** as supplementary material and reference the
   artefact rates when discussing PWC data quality.
5. **Append the reproducibility checklist** as an appendix and define the "minimum viable
   report" standard for future evaluation submissions.
6. **Extract eval_library version** from individual JSON files for a more precise version
   coverage analysis (the flat CSV merges name and version).
