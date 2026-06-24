# Refactoring & Release-Prep Plan — `eee-eval`

**Status:** audit-and-plan only. No source code has been modified in this pass.
**Scope:** prepare the repository for public release — resolve SOLID/maintainability
debt, make the reported numbers reproducible from `data/` + `experiments/`, and
identify files to delete/relocate. Generated from a three-part read-only audit
(src core; analysis/figures/root; reproducibility & release inventory) plus the
data-integrity investigation that found the aggregate is polluted.

Canonical headline numbers (ArXiv = `data/arxiv_extraction_general/llm/` only,
**PWC included**): **37,718 records**, ArXiv 8,387, leaderboard subtotal 28,755,
PWC 576, **6,455 models / 1,293 benchmarks**; §4.1 ArXiv non-null 8,192/8,387 =
97.7%; §4.3 full coverage shots 41.6 / harness 74.1 / CoT 60.5 / prompt 7.0 /
temp 0.9; subset unchanged. These supersede the paper's 43,788 / 14,457.

---

## Part A — Refactors (ordered by leverage)

Sequencing principle: build the shared primitives first (everything else depends
on them), then fix live correctness bugs, then decompose god modules, then
hygiene. Each task is self-contained.

### Tier 0 — Shared primitives (unblock everything)

#### R1. Single EEE record factory
- **Why (DRY/SRP/DIP):** the canonical record shape is `eval_types.EvaluationLog`
  (pydantic), but five emitters hand-assemble it as raw dicts and drift
  independently; two bypass schema validation entirely.
- **Changes:**
  - New `src/utils/eee_record.py`: `build_evaluation_log(model_info, results,
    source_metadata, eval_library, ...) -> EvaluationLog`, serialised via
    `.model_dump(mode="json", exclude_none=True)`.
  - Replace builders in `src/extraction/add_leaderboard_records.py` (`_record`
    L167–226), `hf_model_card_fetcher.py` (`write_model_card_record` L373–475),
    `pwc_fetcher.py` (`_record` L632–730), `extract_paper.py`
    (`make_benchmark_record` L852–972, `make_fallback_record` L975–1027) with
    calls to it.

#### R2. One developer-inference map (data-correctness bug)
- **Why (Single-source-of-truth):** four `get_developer`/`infer_developer`
  implementations **disagree** (`llama→meta-llama` vs `meta`; `qwen→Qwen` vs
  `alibaba`; `deepseek→deepseek-ai` vs `deepseek`). Since developer is part of the
  on-disk path and the conflict key, the same model can be written under two
  folders → phantom cross-source "conflicts" in reported counts.
- **Changes:** keep `src/extraction/constants.infer_developer` as the only
  implementation; delete `src/utils/helpers/developer.py` and
  `src/scrapers/utils.get_developer`; repoint `hf_model_card_fetcher._infer_developer`
  and `pwc_fetcher._infer_developer_from_model` to it. Add a unit test asserting
  canonical org IDs.

#### R3. One HTTP fetch module
- **Why (DRY/SRP):** three `fetch_json/fetch_csv` impls with three error
  contracts (raise vs return `[]`), plus per-scraper `_try_fetch_*` reimplementations.
- **Changes:** keep `src/utils/helpers/fetch.py` (add `safe_*` variants); delete
  `src/scrapers/utils.py` fetchers and every `_try_fetch_*` in
  `alpacaeval2_scraper.py`/`mtbench_scraper.py`/etc.; route
  `add_leaderboard_records.LiveLeaderboardFetcher._get*` through it.

#### R4. One path-component sanitizer
- **Why (DRY):** four sanitizers, three different regexes
  (`convert_eval_logs._write_log`, `scrapers/base._sanitize` L540,
  `add_leaderboard_records._save` L157–159, `extract_paper`).
- **Changes:** add `sanitize_path_component()` to `src/utils/`; replace all four.
  Fix `convert_eval_logs._write_log` return annotation (`-> Path` but returns
  `list[Path]`).

#### R5. `src/analysis/common/` shared analysis primitives (data-correctness)
- **Why (DRY + latent correctness):** normalisation, validity, loading and stats
  are copy-pasted across ~12 analysis scripts with **silent drift** that can make
  different scripts report different published numbers from the same data.
- **Changes:** create
  - `src/analysis/common/normalization.py` — one `normalize_benchmark`,
    `normalize_model_id`, `SOURCE_LABELS`, `is_valid`, `BENCHMARK_FIXUP`.
    Reconcile the divergent maps: `write_collisions.py` drops the
    `humaneval+/mbpp+` rule the others keep (**4a — fix first**); reconcile
    `hfopenllm_v2`↔`open_llm_leaderboard_v2` label key; British/American
    `normalize_model_id` split.
  - `src/analysis/common/loader.py` — one `load_corpus()` encapsulating EEE
    field paths (four record-walkers disagree on shots/harness source and
    `id` vs `name`).
  - `src/analysis/common/stats.py` — `partial_r2_decomposition`, `tau_b_with_ci`,
    `estimate_power`, `eta_squared` (each currently duplicated 2–3× with
    incompatible implementations).
  - Import everywhere; delete the per-file copies in `collision_detection.py`,
    `rerun_statistics.py`, `write_collisions.py`, `aggregate_results.py`,
    `rank_instability.py`, `reanalysis_r2.py`, `variance_decomposition.py`,
    `harness_effect_data.py`, `per_benchmark_ols.py`, `coverage_audit.py`,
    `quality_audit.py`.

### Tier 1 — Live correctness bugs

#### R6. Fix the ArXiv over-count at the source (THE data bug)
- **Why (reproducibility):** `aggregate_results.py` (L104–105) rglobs **all** of
  `data/`, pulling in `arxiv_extraction_general/samples/` (3,311 audit copies),
  `naive/` (363) and duplicate old+new extractions → 50,133-row aggregate.
  Canonical ArXiv = `llm/` only (8,387). `coverage_audit.py` over-counts to
  12,061 the same way; `collision_detection.py`, `rank_instability.py`,
  `reanalysis_r2.py`, `rerun_statistics.py`, `harness_effect_experiment.py`
  inherit it.
- **Changes:** restrict ArXiv ingestion to `data/arxiv_extraction_general/llm/`
  (exclude `samples/`, `naive/`, and the 3 stray top-level jsons) in
  `aggregate_results.py` and `coverage_audit.py`; have every other analysis
  script load via `common.loader.load_corpus()` (R5) so the rule lives in one
  place. Then regenerate all `data/aggregated/*` and `analysis_output/*`.

#### R7. HELM adapter LSP violation (live `TypeError`)
- **Why (LSP):** `HELMAdapter.transform_from_directory` adds a required
  positional `output_path` absent from the base/siblings, so polymorphic calls
  via `convert_eval_logs.py` (`--framework helm/auto`) crash; `_transform_single`
  return annotation also contradicts the base contract.
- **Changes:** match the base signature `(self, dir_path, metadata_args=None)`;
  remove the dead `output_path`/JSONL block (L141–144); fix the return
  annotation to `EvaluationLog`.

#### R8. Remove the always-throwing fetcher from the active registry
- **Why (OCP):** `OpenLLMLeaderboardV1Fetcher._fetch_rows` raises
  unconditionally (L592) yet is registered in `ALL_FETCHERS` (L1402), so every
  full run fails one source by design.
- **Changes:** drop it from `ALL_FETCHERS` (keep the class, documented as
  deprecated) or delete the unreachable body.

#### R9. `run_collision_analysis.py` — hard-coded data + path bug
- **Why (reproducibility/SRP):** the 8 collision pairs driving §5 are a literal
  DataFrame (L27–36); nothing is read from `data/`. Also a `dirname` nesting bug
  resolves outputs to `src/outputs/` instead of repo-root `outputs/`.
- **Changes:** derive the pairs at runtime from `collision_detection.detect_collisions`;
  keep the literal table only as a `tests/` fixture; fix the output path; split
  compute from the `.tex`/`.png`/`.txt` writers.

### Tier 2 — Decompositions (god modules)

#### R10. Decompose `extract_paper.py` (1,295 lines, no class boundaries)
- **Why (SRP/DIP + security):** HTTP, HTML parsing, OpenRouter LLM calls,
  validation, record building and CSV/summary I/O are one flat namespace;
  `verify=False` disables TLS (L312, L770–786).
- **Changes:** extract `ArxivHtmlClient` (fetch + section extraction),
  `LLMClient` interface + `OpenRouterClient` impl in `src/utils/` (inject; remove
  `verify=False` or gate behind a documented opt-in), `HaikuExtractor`,
  `MetricResolver`/`RelationshipResolver`/`HarnessResolver` (from the
  `make_benchmark_record` god function), `RecordBuilder` (delegates to R1),
  `SummaryWriter`. `main` becomes wiring only. Replace the
  `_already_exists._cache` function-attribute global with a `RecordExistenceIndex`
  class. Unify the second OpenRouter client in `pwc_fetcher.py` onto the same
  `LLMClient`.

#### R11. Converter base-class shared envelope
- **Why (DRY/OCP):** the three adapters independently build `SourceMetadata`,
  `EvalLibrary`, `evaluation_id`, timestamps, and the instance-level wiring
  (HELM ≈ Inspect copy-paste).
- **Changes:** add `_build_source_metadata`, `_build_eval_library`,
  `_make_evaluation_id`, `_attach_instance_level` to
  `src/converters/common/adapter.BaseEvaluationAdapter`; subclasses keep only
  library-specific result parsing.

#### R12. Scraper CLI/base consolidation
- **Why (DRY/OCP/encapsulation):** six scrapers duplicate `main()`, the
  `sys.path` hack (one path is wrong — `utils/` not `src/utils`), `_convert_row`,
  fallback timestamping, and call the **private** `_write_records` instead of the
  public `BaseLeaderboardScraper.run_standalone` that exists for this.
- **Changes:** add `BaseLeaderboardScraper.cli()` (generates argparse `main`
  once); route all scrapers through `run_standalone`; delete per-file `main` +
  direct `_write_records`; fix the bogus path insert; lift the `_FALLBACK_TS`
  pattern into the base.

#### R13. Decompose analysis/figure god `main()`s
- **Why (SRP):** `run_analysis.py` `main()` (L32–776) and `rerun_statistics.py`
  `main()` (~1,200 lines, all helpers nested) and `generate_all_figures.py`
  (1,172 lines) interleave load + stats + LaTeX + plots; `reproduce_case_studies.py`
  / `generate_evidence_file.py` inline every case study.
- **Changes:** one function per task/section/case-study/figure returning
  structured results; `main` = load → orchestrate → emit. Promote nested helpers
  to module scope (use R5). Separate computation from `.tex`/`.png`/`.md`
  emission. Remove dead blocks (`rerun_statistics.py` `df_old` no-op deltas
  L243/333–336/1199–1203; dead groupby L1109–1111).

#### R14. Externalise paper constants to one source of truth
- **Why (reproducibility/DIP):** "expected/claimed" paper numbers are baked into
  `final_audit.py`, `reproduce_case_studies.py`, `generate_evidence_file.py`,
  `run_collision_analysis.py`, and several figure scripts (`macros.tex`
  `\totalrecords{29,331}`, `coverage_bars.py` "49,420"). When data is
  re-extracted, audits emit false PASS/FAIL.
- **Changes:** create `paper_claims.json` (or read `submission/latex/macros.tex`)
  as the single source; `final_audit.py` / `reproduce_case_studies.py` /
  `verify_paper_claims.py` read it; figure scripts read coverage/collision values
  from `analysis_output/*.csv` at runtime; `final_audit.py` references the
  already-computed `mean_abs_delta` instead of re-typing `0.14`.

### Tier 3 — Consolidation & hygiene

#### R15. Consolidate validation
- **Why (DRY/SRP):** `src/validation/schema.py` and `validate_all.py` implement
  the same `AutoFixer`/`FileValidator`/`BatchValidator`; `quality_audit.py`
  re-implements schema + coverage with different rules, so auditor and fixer
  disagree on out-of-range scores and benchmark canonicalisation.
- **Changes:** make `validate_all.py` a thin wrapper over `schema.py` (fold in its
  only extra: `fixable→exit 2`); share one `is_score_out_of_range()` and
  `normalize_benchmark()` (R5) between auditor and fixer.

#### R16. Retire/segregate the dead figure tree
- **Why (dead code/DUP):** the paper includes only `coverage_bars.pdf`,
  `case1_parser_flow_*`, `case_studies_unified_*` (from
  `LLM_Evaluation_Report/figures/scripts/`); the ~40 figures in `src/figures/`
  are unreferenced and include 3+ rival coverage-bar impls that disagree.
- **Changes:** treat the 3 report scripts as canonical; move `src/figures/` to
  `experiments/exploratory_figures/` or delete; fix CLAUDE.md "Key entry points".

#### R17. Logging, paths, and `src/utils` cohesion
- **Why (maintainability/DIP):** mixed `print` vs `logging`; blanket
  `except Exception: pass` that silently drops records (a completeness risk);
  `preflight_check.py` references a stale `scripts/scrapers/raw` layout and
  **mkdir's** missing dirs inside a "check"; `final_audit.py` uses
  `ROOT=Path('.')`; per-source adapters live under `src/utils/` (incl.
  `global-mmlu-lite/` whose hyphen makes it unimportable).
- **Changes:** one `get_logger()` in `src/utils/`; replace `print`/bare excepts
  (log + count drops); use `Path(__file__).resolve().parents[N]` everywhere;
  remove mkdir side-effects from preflight; move leaderboard adapters out of
  `src/utils/` (to `src/scrapers/` or `src/adapters/`), keep `utils/` for shared
  helpers only; rename/remove `global-mmlu-lite`.

---

## Part B — Reproducibility tasks

1. **Pin dependencies + ship a lockfile.** `pyproject.toml` uses floors only
   (`numpy>=1.24`, …) and there is **no lockfile** and **no `requirements.txt`**
   (README's `pip install -r requirements.txt` is broken). Add upper bounds /
   exact pins, commit `uv.lock` (currently gitignored), and generate
   `requirements.txt`. Align `.python-version` (3.13) with
   `requires-python` (>=3.11) — pick one.
2. **Document env / secrets.** Required: `OPENROUTER_API_KEY`, `HF_TOKEN`
   (the latter missing from `.env.example`). Add `HF_TOKEN` (+ optional
   `REQUESTS_CA_BUNDLE`) to `.env.example`; never ship `.env`.
3. **Seeds & determinism.** RNG is already seeded everywhere
   (`np.random.default_rng(42)`) — keep. Remove the baked `Timestamp.now()`
   header written into `analysis_output/collision.txt` by `rerun_statistics.py`;
   add `sorted()` to the FS-order-dependent within-source dedup
   (`grp.iloc[0]` on an unsorted rglob picks an order-dependent representative).
   Pin the LLM to a dated snapshot instead of the floating
   `anthropic/claude-haiku-4.5` alias (extract_paper.py, pwc_fetcher.py).
4. **Resolve output-file collisions.** `collision.txt` is written by both
   `rerun_statistics.py` and `write_collisions.py`; `collision_source_pairs.csv`
   by both `collision_detection.py` and `collision_overlap_matrix.py` (different
   schemas) — pick one producer each (last-run-wins today).
5. **Single source of truth for headline numbers.** Paper, README, CLAUDE.md,
   `macros.tex`, `coverage_bars.py` ("49,420"), `fig8_coverage_projection.py`
   (`n_sources_current=71`) all disagree. Adopt the canonical recompute
   (37,718 / ArXiv 8,387 / 6,455 models / 1,293 benchmarks; §4.3 as above) and
   propagate from `analysis_output/recompute_canonical.txt`. Delete the stale
   `analysis_output/dataset_stats_and_case_studies.md` (50,518-row May snapshot).
6. **One command to regenerate the analysis numbers.** There is none today
   (README points to a non-existent `reproduce_paper.py`; `reproduce_case_studies.py`
   covers only §5). After R6, add `reproduce_paper.py` that runs, from `data/` +
   `experiments/`: `aggregate_results` → `coverage_audit` → `collision_detection`
   → `run_analysis`/`run_collision_analysis` → figure scripts → `final_audit`,
   writing all `analysis_output/*` and `outputs/*` deterministically. Document it
   as the single repro entry point (dataset *rebuild* from network/LLM remains
   non-reproducible and should be documented as such, separate from analysis
   repro).
7. **Capture environment.** Add a short `ENVIRONMENT.md` (or extend README)
   recording Python version, OS, `uv.lock` hash, and that GPU experiments
   (E3 harness-version, E6 BBH, factorial grid) were run on Colab A100 with
   `lm-evaluation-harness` 0.4.3/0.4.11 + vLLM (versions pinned per experiment).

---

## Part C — Verification plan (behaviour-preservation)

**Existing tests (10 files):** integration — `test_helm_adapter`,
`test_helm_instance_level_adapter`, `test_inspect_adapter`,
`test_inspect_instance_level_adapter`, `test_lm_eval_adapter`, `test_scrapers`;
unit — `test_benchmark_canonicalization`, `test_check_duplicate_entries`,
`test_extract_paper`, `test_validation`.

**Coverage gaps (no tests today):** all of `src/analysis/` (collision, coverage,
aggregate, variance, rank-instability, power), all figure scripts, the root
scripts (`reproduce_case_studies.py`, `generate_evidence_file.py`,
`verify_paper_claims.py`), and the shared primitives to be created.

**Before refactoring — establish baselines (do first):**
- Freeze a golden snapshot of every `analysis_output/*.csv/.txt` and `outputs/*`
  generated from the **canonical** corpus (after R6), checked into
  `tests/golden/`. Refactors R5/R9/R13/R14 must reproduce these byte-for-byte
  (modulo the intentionally-removed timestamp).

**Per-refactor verification:**
| Refactor | How verified | Gap to fill |
|---|---|---|
| R1 record factory | adapter/scraper integration tests + `test_extract_paper`; assert emitted JSON validates against `eval.schema.json` | add schema-validation assertion to factory unit test |
| R2 developer map | new unit test of canonical org IDs; diff `data/` paths before/after on a fixed sample | none once test added |
| R3 fetch / R4 sanitizer | `test_scrapers` (mock HTTP); unit test sanitizer table | add sanitizer unit test |
| R5 analysis common | golden-file diff of all `analysis_output/*`; `test_benchmark_canonicalization` extended | add `loader`/`stats` unit tests |
| R6 ArXiv over-count | assert `aggregate_results` ArXiv count == 8,387 and total == 37,718; golden diff | add count assertion test |
| R7 HELM LSP | `test_helm_adapter` + new test driving all 3 adapters through `convert_eval_logs.main(--framework auto)` | add the cross-adapter integration test |
| R8 dead fetcher | `test_scrapers` run-all asserts no source raises | extend |
| R9/R13/R14 god-script splits | golden-file diff of `outputs/*` and `.tex`; extracted functions get unit tests | add analysis/figure regression tests |
| R10 extract_paper split | `test_extract_paper` (extend with injected fake `LLMClient`/`ArxivHtmlClient`) | add fakes/fixtures |
| R11 converter base / R12 scraper base | existing adapter + scraper integration tests cover these directly | none |
| R15 validation | `test_validation` extended to both entry points | extend |
| R16 figures | visual/CSV-input diff for the 3 canonical scripts only | manual |

**General gate:** `pytest` green + golden-output diff clean after each tier;
run `reproduce_paper.py` (R6 of Part B) and confirm it reproduces the frozen
baseline.

---

## Part D — Release cleanup (delete / relocate — execute in a later pass)

**Must NOT ship (secrets / large blobs):**
- `.env` (real API keys) — exclude from any archive/upload.
- `data.zip` (~36 MB) — data belongs on HF Hub per `.gitignore`, not the release tree.

**Regenerate, don't ship (stale generated artifacts):**
- `data/aggregated/all_results.csv`, `coverage_stats.json`,
  `pipeline_coverage_report.json` (polluted — rebuild after R6).
- `analysis_output/*` in full (over-counted), especially
  `dataset_stats_and_case_studies.md` (50,518 snapshot — delete).
- root `section5_evidence.txt`, `VALIDATION_REPORT.json` (regenerate).

**Relocate out of the aggregation path (cause the over-count):**
- `data/arxiv_extraction_general/samples/` (3,311 — audit copies; keep for the
  §4.1/§4.2 audits but move outside `data/` or hard-restrict aggregation to `llm/`).
- `data/arxiv_extraction_general/naive/` (363 — superseded by `llm/`).

**Strip (build/agent/IDE cruft):** all `__pycache__/` + `.pyc`,
`experiments/.claude/`, `.claude/`, `.vscode/`, `.pytest_cache/`, `.cache/`.

**Experiment WIP (decide keep vs ship):**
`experiments/harness_reeval/version_compare/` (untracked WIP for the
harness-version E3/E6 work) — its 27 `.log` files (~2 MB) and ad-hoc launchers
(`_run_*.sh`, `orchestrate.sh`, `auth_start.sh`, `make_html.sh`) should not ship
as-is; promote the runner + `compare_versions.py` into a documented experiment
and drop the logs/launchers.

**Doc/layout drift to fix before release:**
- README describes a Docling pipeline that no longer exists (`src/extraction/pipeline.py`,
  `docling_parser.py`, `table_parser.py`, `converter.py`, `llm_fallback.py`,
  `protocol.py`, `download.py`, `prose.py`) and references missing
  `requirements.txt` / `reproduce_paper.py` / `src/inspect_adapter/` / `evalspec/`.
  Rewrite install + reproduce sections to match actual files.
- CLAUDE.md lists `count_papers.py` as a load-bearing entry point — it does not
  exist. Fix or remove.
- Empty placeholder dirs at root (`figures/`, `scripts/scrapers/`) — remove.
- E-number drift: scripts label the scoring-mode experiment "E5"/"E3" while the
  paper uses E2 — reconcile.

---

## Suggested execution order across passes
1. **R6 + Part B(5,6)** — fix the aggregate, regenerate, freeze golden baselines,
   reconcile headline numbers. (Highest value; unblocks honest verification.)
2. **Tier 0 (R1–R5)** — shared primitives.
3. **Tier 1 (R7–R9)** — live bugs.
4. **Tier 2 (R10–R14)** — decompositions.
5. **Tier 3 (R15–R17)** + Part D cleanup + README/CLAUDE rewrite.
