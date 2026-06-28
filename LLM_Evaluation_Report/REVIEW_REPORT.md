# REVIEW_REPORT.md — EACL/ARR review-prep for `acl_latex.tex`

Senior-reviewer + co-author pass over `latex/acl_latex.tex` ("How To Report
Evaluations of your LLM…"). Adversarial review, conservative editing. Numbers are
never altered — inconsistencies are flagged, not "corrected."

**Environment constraints (decided with author 2026-06-27):**
- No LaTeX toolchain on this machine → **compilation skipped**. Edits are made by
  inspection with a per-pass brace/`\ref` sanity check. Anything build-risky is
  flagged below; the author compiles in Overleaf.
- `figures/eval_record_attrition_chain.pdf` (Fig. `fig:chain`) is **missing** with
  no generator → **float commented out**, caption preserved, the two
  `\ref{fig:chain}` mentions rewired to prose.

Venue target: **EACL via ARR**. Confirmed against live CfP: long papers **≤ 8
pages content**; references + appendices + **mandatory Limitations** do not count;
double-blind; AI-assistance disclosure required.
([EACL 2026 CfP](https://2026.eacl.org/calls/papers/),
[EACL 2027 CfP](https://2027.eacl.org/calls/papers/))

---

## Status legend
✅ fixed · ✍️ written · ⚠️ flagged (author decision) · 🔎 verified-consistent · ⏭️ deferred

---

## Pass 0 — Reconnaissance (done)
- Read full paper (1785 lines), 3 skills, README update, bib, figures, CfP.
- Figure inventory: `coverage_bars`, `case1_parser_flow_S3`,
  `case_studies_unified_belebele_S3`, `case_studies_unified_hellaswag_S3` **present**;
  `eval_record_attrition_chain.pdf` **absent** (handled, see Pass 1).
- All live `\ref`s resolve except the two intentionally rewired `fig:chain` refs.
- README numbers cross-check the paper's canonical figures (46,559 / 7,398 / 1,954;
  97.5% precision; coverage table) → ✅ consistent.

---

## Findings & resolutions

### BLOCKERS
| # | Finding | Loc | Status |
|---|---------|-----|--------|
| B1 | Abstract is a placeholder ("Abstract to be added.") | L112 | ✍️ written (Pass 1) |
| B2 | Conclusion is a placeholder ("[To be written last…]") | L1247 | ✍️ written (Pass 1) |
| B3 | No Limitations section (ARR desk-reject risk) | after L1247 | ✍️ written (Pass 1) |
| B4 | Referenced figure `eval_record_attrition_chain.pdf` missing, no generator | L973 | ✅ float commented + refs rewired (Pass 1) |
| B5 | No AI-assistance disclosure (ARR requirement) | — | ✍️ added (Pass 1) |

### CONSISTENCY / ARGUMENT INTEGRITY
| # | Finding | Loc | Status |
|---|---------|-----|--------|
| C1 | Temperature framed as "most responsible for score divergence" then shown <1% variance | L190–196 | ✅ reframed (Pass 1) |
| C2 | `scoring_mode` is the headline EvalSpec field but never appears in the coverage audit | §4.3 | ✅ added explanatory paragraph (Pass 1) |
| C3 | Headline example Gemma-7B vs Fig/attrition Gemma-2-9B handoff | L177–178 | ✅ made explicit in prose during fig rewire (Pass 1) |
| C4 | Overclaim "no leaderboard API exposes the evaluation harness at all" vs §4.3 "none we examined" | L192 vs L608 | ✅ unified to "none…we examined" (Pass 1) |
| C5 | `59.60 "BigBench-Hard"` row reads as a pipeline bug inside a results table | L723–724, tab:bbh-app | ⚠️ footnoted cleanly; alias-fix flagged for author (Pass 1) |
| C6 | Underpowered MMLU/GPQA claims must read "directional, not confirmatory" | §5, App G | 🔎 already phrased correctly; verified (Pass 2) |
| C7 | "100 ArXiv papers → 1,726 models / 1,806 benchmarks" needs explanation | §3.2 | ✅ added clause (Pass 1) |

### CITATIONS
| # | Finding | Loc | Status |
|---|---------|-----|--------|
| R1 | HellaSwag introduced with no citation | L898 | ✅ added `\citep{zellers2019hellaswag}` + bib entry (Pass 1) |
| R2 | Active `\cite{}` mixed with natbib `\citet/\citep` | L705,742,820,862 | ✅ standardized (Pass 1) |
| R3 | Dangling `\ref{sec:case-gpqa}` | L1188 | 🔎 inside `%`-commented block → dead, not live. No action needed |
| R4 | Related work: cite HELM (Liang et al.) + lm-eval-harness (Gao et al.)? | §2 | ✅ added (Pass 2) — verified entries, wired into the EEE-schema "HELM–EleutherAI gap" sentence |
| C8 | Discussion claimed Fig. coverage shows harness "zero valid coverage" — contradicts the 93.2% pipeline-attached value in the same figure | §6 (was L1122) | ✅ fixed (Pass 2): harness separated from the genuinely-zero pair |
| W1 | Comma splices + jargon ("as built") in intro/related work | L201, L289–290, L300 | ✅ fixed (Pass 2) |

### NUMBER HYGIENE (verify only — values unchanged)
| # | Check | Result |
|---|-------|--------|
| N1 | η² GSM8K format 27% (body) vs 27.4% (App) | 🔎 consistent (body rounds) |
| N2 | η² n-shot 65% (body) vs 65.1% (App) | 🔎 consistent |
| N3 | 306/312 grid counts | 🔎 consistent (312 attempted, 306 completed) everywhere |
| N4 | Math-Verify +4.66pp / 3,751 models / ~90pt, attributed to kydlicek2025mathverify | 🔎 consistent |

---

## AUTHOR DECISION NEEDED
1. **Title.** Current — *"How To Report Evaluations of your LLM: A Story of
   LLM/Human Misunderstandings"* — undersells a cross-source metadata-attrition
   paper. Proposed alternatives (pick one or keep current):
   - A. *Why LLM Leaderboards Are Incomparable: Structural Fragmentation and Missing Evaluation Metadata at Scale* (matches the companion-paper framing/README)
   - B. *The Score Without the Procedure: Metadata Attrition in Public LLM Evaluation*
   - C. *Grounding an Evaluation-Reporting Standard in a Cross-Source Metadata Audit (EvalSpec v0.1)*
2. **`59.60` BigBench-Hard alias (C5).** I footnoted it (value preserved, excluded
   from automated collision counts). The *real* fix is to add the "BigBench-Hard"
   alias to the normaliser so it counts properly, or to name the source paper in
   `tab:bbh-app` (currently listed only as "(BigBench-Hard alias)"). Your call.
3. **HELM + lm-eval-harness citations (R4).** Worth adding to §2 for reviewers who
   will ask "why another spec / what about HELM." Not in `custom.bib`. Proposed
   verified entries are staged below — say the word and I'll wire them in.
4. **`eval_record_attrition_chain.pdf` (B4).** Float is commented out to compile.
   Restore the figure (or accept its removal — the SEA-LION case stands in prose
   without it) and I'll un-comment + re-add the `\ref`s.
5. **Page count.** Cannot measure without compiling. The added Abstract +
   Conclusion add to the ≤8pp content budget (Limitations + refs + appendices are
   exempt). Verify in Overleaf; if over, the Discussion/Recommendations prose has
   the most slack.

### Proposed bib entries for R4 (staged, not yet inserted)
```bibtex
@article{liang2023helm,
  title={Holistic Evaluation of Language Models},
  author={Liang, Percy and Bommasani, Rishi and Lee, Tony and others},
  journal={Transactions on Machine Learning Research},
  year={2023}
}
@misc{gao2024lmevalharness,
  title={The Language Model Evaluation Harness},
  author={Gao, Leo and Tow, Jonathan and Abbasi, Baber and others},
  year={2024},
  howpublished={\url{https://github.com/EleutherAI/lm-evaluation-harness}}
}
```

---

## Change log (per pass)
- **Pass 0:** reconnaissance only — no edits.
- **Pass 1 (blockers + safe consistency/citation fixes):**
  - `acl_latex.tex` abstract written (B1); conclusion written (B2); **Limitations**
    section added (B3); **Use of AI Assistants** disclosure added (B5).
  - Missing-figure float wrapped in `\iffalse…\fi` + TODO (B4); both `\ref{fig:chain}`
    rewired to prose, Gemma-7B→Gemma-2-9B handoff made explicit (C3).
  - Intro temperature reframed + harness overclaim softened to "none…we examined" (C1/C4).
  - `scoring_mode`-not-in-coverage paragraph added to §4.3 (C2).
  - `59.60` BigBench-Hard inline admission → clean footnote (C5).
  - ArXiv 1,726/1,806 explanation added to §3.2 (C7).
  - HellaSwag `\citep{zellers2019hellaswag}` + verified bib entry (R1).
  - 4 active `\cite{}` → `\citet/\citep` (R2).
  - Verified: no live dangling refs; `\iffalse/\fi` balanced; no active `\cite{}` left.
  - **Build not run** (no local LaTeX, per author decision) — needs an Overleaf compile.
- **Pass 2 (writing-polish + HELM/Gao citations):**
  - Added verified `liang2023helm` (HELM, TMLR 2023) and `gao2024lmevalharness`
    (EleutherAI lm-eval-harness, Zenodo concept DOI) to `custom.bib`; wired both
    into the EEE-schema "HELM–EleutherAI gap" sentence (R4). Non-redundant with the
    existing `biderman2024` "Lessons from the Trenches" harness-methodology cite.
  - Fixed harness "zero valid coverage" contradiction in §Discussion (C8).
  - Fixed three comma splices + removed "as built" jargon (W1).
  - Verified: both new keys defined+used; sections in order
    (Conclusions→Limitations→AI-disclosure→Acknowledgments→References); no active `\cite{}`.
- **Pass 3 (figures/tables + final consistency sweep):**
  - **Consistency sweep — zero numeric contradictions found.** Every dataset count
    (46,559 / 28,755 / 17,228 / 26,790 / 7,398 / 1,954 / 5,301 / 1,726 / 1,806 / 576)
    reconciles across abstract/intro/§3.2/tables/figure-captions/conclusion. Arithmetic
    verified: the 10 leaderboard rows sum to exactly 28,755; 28,755+17,228+576=46,559.
    Audit numbers consistent: 1,977/2,027=97.5% (residual 50), 712/817=87.1%, 869/869=100%,
    312 attempted/306 completed. η² body-rounding (27/65/93%) vs appendix-exact
    (27.4/65.1/93.2%) is intentional and standard. No stray 2,029 / 967 / 86.2 in the paper.
  - **Figures/tables:**
    - Fixed 2 **unreferenced figures**: `fig:casestudies-unified` (main §5 figure, was
      only referenced from the appendix) now introduced in the §5 opening; `fig:case1-parser`
      (referenced nowhere) now cited in §5.1 (`sec:gsm8k`).
    - Fixed 2 **unreferenced appendix tables**: `tab:e1-gsm8k`, `tab:e2-scoring-mode`.
    - Added a body pointer to Appendix `app:hellaswag`.
    - `tab:bbh-app` caption made self-contained re: the "BigBench-Hard" alias row.
    - All 4 live `\includegraphics` files exist; the 5th is the `\iffalse`-guarded chain.
    - `\FloatBarrier` placement sane (2 in appendix). **No dangling refs** (nothing → `??`).
  - Remaining orphan labels are harmless: `fig:chain` (commented), `sec:*` (unused
    section anchors, only referenced in the commented "paper organisation" paragraph).
- **Pass 4 (prose read-through / AI-tell lens + structural balance):**
  - Scanned the full body for AI-tell vocabulary (leverage/delve/crucial/robust/
    notably/underscore/…) and "not only…but" negative parallelisms: **none found**.
    The author's prose has genuine voice; the new sections match register. No forced edits.
  - One stylistic observation (not changed): em-dash density is high (378 `---`), but
    it is the author's consistent voice throughout — left intact. Flag for the author
    if they want to thin them for camera-ready.
  - Structural balance verified (cannot compile here): table 13/13, table\* 1/1,
    figure 2/2, figure\* 3/3, tabular 14/14, itemize 5/5, document/abstract 1/1;
    real `\iffalse`/`\fi` matched. Document is structurally sound.

---

- **Snapshot desync resolved (2026-06-28):** A revision pack arrived asserting
  49,420 / 11 sources / ArXiv 20,665. Investigated: those numbers exist only in
  `acl_lualatex.tex` (abandoned, last real edit June 13) and the stale `-5.pdf`
  built from it; the CHANGELOG shows 49,420 was *superseded* at Pass 18.
  `analysis_output/paper_numbers.json` (generated from data), `CLAUDE.md`, memory,
  and the fresh `-6.pdf` (built from `acl_latex.tex`) all confirm **46,559 / 12
  sources**. Author confirmed: **stay on 46,559, ship `acl_latex.tex`.** The pack's
  ready-to-paste sections (49,420) are NOT used.
- **Pass 5 (substantive revision, S1–S9 punch-list, on 46,559):**
  - S1 — dominant-source reframe added to §4.3 ("one platform plus a sparse tail",
    93.2% owned); OLv2-ablated coverage row parked for author (uncomputed).
  - S3 — **EvalSpec worked example** added to §6: new `tab:evalspec-worked`
    instantiating EvalSpec-R for the SEA-LION/OLv2 collision, showing `scoring_mode`
    + `harness_name`+`eval_timestamp` resolve it. Converts EvalSpec proposal→demo.
  - S4 — explicit case-studies↔factorial-grid bridge added to §5 opening.
  - S5 — "audit-grounded position paper" framing committed in the intro.
  - S9 — explicit machine-validated + audit-grounded differentiation sentence in §2.
  - S2 (recall), S7 (temperature demotion), S8 (parser-artifact framing): **verified
    already present** in the current file — not redone. S6 (title): parked.
  - Verified: `tab:evalspec-worked` labeled+referenced; tabular 15/15, table 14/14;
    no dangling refs; 46,559 intact, zero `49,420`.
- **Pass 6 (style reframe toward exemplar `alzahrani2024`):**
  - **Echo-check CLEAN** — none of the exemplar's signature phrases ("taken at face
    value", "when benchmarks are targets", Kendall/RStd, "8 positions") appear; no
    content/structure/metric bleed. **Zero hype words** in the paper.
  - Finding: the paper's voice already matches the exemplar (plain declarative,
    active "we", stakes-first Gemma-7B hook, numbered findings). Main gap = em-dash
    density (384). Thinned the abstract to 0 dashes + one intro spot; the full
    384→~75 sweep is parked as optional (mechanical; risks over-editing the voice).

- **Pass 7 (academic-paper revision mode + Style Calibration to `alzahrani2024`):**
  - **Phantom em-dash finding:** the "~378 em-dashes" was an artifact of counting
    commented-out `%% DRAFT` blocks (345 of them). The **live** count is **35**
    (29 in body) — normal ACL density, already near the exemplar. No aggressive
    thinning warranted; the Style Spec's "378→75" premise was counting dead prose.
  - Edits: converted the §4.1 \emph{Precision}/\emph{Recall} em-dash definitions to
    plain declaratives that define each metric operationally before use (Style Spec:
    "plain declarative" + "define metrics operationally"). Live em-dashes 35→31.
  - **Writing Quality Check: pass.** No AI-typical terms, no hype words, no
    throat-clearing openers, varied paragraph rhythm.
  - **Echo-check vs exemplar: clean** (re-confirmed) — zero signature-phrase overlap.
  - Verdict: the manuscript already sits in the exemplar's register (plain
    declarative, active "we", stakes-first hook, numbered findings). Remaining
    optional hygiene: delete the superseded `%% DRAFT` comment blocks (zero PDF
    impact; removes the phantom em-dash count) — offered, not yet done.

- **Pass 8 (reviewer response — computed ablation + schema attribution):**
  - **S1 CLOSED with real data.** Computed the OLv2-excluded coverage ablation from
    `analysis_output/coverage_stats.csv` (method reproduces the published subset
    numbers exactly: 98.0/93.2/77.7). 9-source tail (1,965 records): **n_shot 71.0%,
    harness/prompt_template/temperature/chain_of_thought all 0.0%.** Added as a 3rd
    column to `tab:coverage-app` + cited in §4.3 + Limitations. Reviewer ablation ask
    answered with verified numbers, no fabrication.
  - **Schema attribution (user instruction):** §2 now states the paper "is built on,
    and extends, the open Every Eval Ever (EEE) schema" with a
    `github.com/evaleval/every_eval_ever` footnote, and the EEE paragraph closes on
    releasing dataset+EvalSpec "so the community can adopt, critique, and extend them."
  - Verified: coverage table 4-col alignment OK, tabular 15/15, 71.0% consistent x3.

- **Pass 9 (reviewer-response batch, autonomous):**
  - **Item 1 — `scoring_mode` controlled vocabulary** specified in the EvalSpec
    appendix: precise defs of log-likelihood/generation/preference/execution +
    decision rules for hybrids (rank-by-LL-then-generate, preference-judge with
    short-answer extraction, CoT+MC post-check).
  - **Item 2 — related work** (§2): new paragraph positioning vs HELM scenario
    metadata/per-run logs and the NeurIPS/ML reproducibility checklist
    (`pineau2021reproducibility`, verified JMLR v22 / arXiv:2003.12206 — added to
    bib), plus a concrete SHACL example (judge_model non-null when
    scoring_mode=preference) catching the TruthfulQA/judge omission.
  - **Item 3 — prevalence caveat** (Limitations "Existence, not prevalence"):
    case studies show existence/mechanism, not ecosystem prevalence; census left
    to future work via the collision pipeline.
  - **Item 4 — new EvalSpec fields**: added `judge_prompt_hash`,
    `answer_parser_version` to Recommended tier + §6 prose; specified hash
    canonicalization (strip/collapse whitespace, normalize placeholders).
  - **η² bootstrap (real computation)**: from `controlled_eval_results.jsonl`,
    10,000 cell-resamples. GSM8K format η²=27.4%, **95% CI [14.1%, 44.8%]** (robust,
    excludes small effects → not a limit=200 artefact); MMLU η²=24.5%, **95% CI
    [3.3%, 85.4%]** (very wide → quantifies the underpowering). Point estimates
    match the paper exactly. Added to the Factorial-Grid appendix. Script:
    `scratchpad/boot_eta.py`.
  - **Future work**: expanded judge study (more judge families/tasks) added to
    Limitations.
  - **Skipped per author:** SEA-LION author-contact item; release-roadmap
    (prompt-template release / EvalSpec-CI / gold subset).
  - Verified: pineau cite def+used; EvalSpec table alignment OK; table/tabular
    14/14 & 15/15; no dangling refs; bootstrap CIs + new fields present.
  - ⚠️ **Page budget:** body grew (the §2 related-work paragraph and §6 new-field
    paragraph are the main additions; everything else is appendix/Limitations =
    page-exempt). Verify ≤8pp on compile; those two paragraphs are the trim targets.

- **Pass 10 (prompt-template release + per-source "tail" ablation):**
  - **Per-source coverage table** (`tab:coverage-persource`) added to the coverage
    appendix from `coverage_stats.csv`: all 10 leaderboard sources, OLv2 separated
    from the 9-source tail. Shows only OLv2 has non-zero harness (100, pipeline-
    attached) / CoT (83.4); the tail is uniformly 0.0% on harness/prompt/temp/CoT
    with only protocol-fixed n-shot present. This is the per-source view behind the
    aggregate ablation already in `tab:coverage-app`.
  - **Prompt templates released (#1).** Extracted the *actual* rendered prompts from
    the saved `samples_*.jsonl` (ground truth) — NOT from `run.py`, which uses
    different format names (standard/cot/fewshot + "Let's think step by step.") that
    do **not** match the runs; releasing from it would have been wrong. Real formats
    differ only by system prompt (plain=model default; instruct/cot=fixed, identical
    across all 4 models). Wrote `experiments/controlled_eval/prompt_templates.md`
    (full worked example per format) and added appendix subsection `app:prompts`
    (referenced from the Factorial-Grid description). No compute needed.
  - Verified: per-source table 7-col aligned; table/tabular 15/15 & 16/16; no
    dangling refs; both new labels referenced; supplementary file written (11 KB).

- **Pass 11 (compaction to the 8-page limit, exemplar style):**
  - Body text **8,228 → 6,768 words** (~1.5 pages); 1 figure + 2 tables relocated
    to the page-exempt appendix (~0.7 page). Est. ~2 pages of body saved.
  - **Lossless moves:** deleted body `tab:e1-body` (dup of appendix `tab:e1-gsm8k`);
    moved `case1_parser` figure → appendix E1; moved `tab:evalspec-worked` → appendix.
  - **Condensed (all numbers/citations/labels preserved):** Case Study 1 and 2
    (7 `\subsubsection` + 3 itemize blocks → tight `\paragraph` findings); the §5
    factorial-grid intro (design detail is in the appendix; added the bootstrap CI);
    the Discussion "Misunderstanding Chain" (its 1st paragraph repeated the intro;
    audit numbers live in §4).
  - Verified: all environments balanced; no dangling refs; every case-study number
    present. **Cannot measure exact page count without a compile** — estimate ~7.5–8
    body pages now. If still over after your Overleaf build, next trim targets:
    EvalSpec field-selection prose, Recommendations, §4.3, and deleting the
    superseded `%% DRAFT` comment blocks (cosmetic, no page effect).

- **Pass 12 (further compaction for 8-page margin):**
  - Condensed EvalSpec field-selection, the three Recommendations, and the score-
    and harness-identification audit sections (granular error/per-paper breakdowns
    tightened, all counts kept). Body **6,768 → 6,321 words**.
  - Cumulative this session: body **8,228 → 6,321 words** (~1.9 pages text) + 1
    figure & 2 tables moved to appendix (~0.7 page) = **~2.5 body pages cut**,
    fully lossless. Est. body now ~7.3 pages (margin under the 8-page limit).
  - Verified: balanced; no dangling refs; all audit/case-study numbers preserved.
  - **Still requires an Overleaf compile to confirm the exact page count.**

- **Pass 13 (compaction to ~5,500-word target, exemplar-style):**
  - Studied the exemplar's compaction: findings stated in section titles, short/late
    related work, detail in appendix. Applied: compressed Related Work (~400→240 w),
    the EEE-schema field listing, both Materials subsections, §4.3 coverage prose
    (merged the redundant headline/dominant-source overlap), and Case Study 3 — which
    also lost the last body `\subsubsection` and gained a finding-style title
    ("Shared Pipelines Fake Independence").
  - **Body now 5,623 words** (session: 8,228 → 5,623, ~32% cut) + 1 figure & 2 tables
    in appendix. Est. ~7 body pages, comfortably under 8.
  - Verified: all balanced; no dangling refs; body `\subsubsection`-free; every
    headline number (46,559/28,755/97.5/87.1/71.0/27%/0.28/1.35/34.10/15pt/93.2) and
    every citation preserved.
  - **Still needs an Overleaf compile to confirm the exact page count.**

- **Pass 14 (EvalSpec reframe + AI-section removal + further cut):**
  - **Removed the "Use of AI Assistants" section** (per author). NOTE: ARR requires an
    AI-assistance disclosure; the methodological LLM use is still described in §3.1,
    but the explicit disclosure statement is now gone — confirm it's covered in the
    Responsible-NLP checklist before submission.
  - **EvalSpec reframed**: no longer "proposed v0.1 specification"; now framed as an
    **adaptation of the open EEE schema** ("EvalSpec adapts the open EEE schema…"),
    released with the dataset so future papers report extractable scores and the
    community can extend it. **All "v0.1" / version language removed** (abstract,
    intro, §2, §6 title, conclusion, appendix). Tier names (Required/Recommended/Full,
    EvalSpec-R) kept — those are tiers, not versions.
  - Compressed the EvalSpec encoding/instrumentation/worked-example, the intro
    chain-of-hands paragraph, and the scoring-mode paragraph.
  - **Preserved** the author's newly-added expanded judge experiment (8 judges, 28
    pairs, 8 task categories) in the Limitations zone.
  - **Word counts:** counting body (Intro→Conclusions, the 8-page-limit scope) =
    **4,844 words**; exempt Limitations = 519; total pre-appendix = 5,363. The
    counting body is comfortably under 8 pages.
  - Verified: balanced; no dangling refs; headline numbers intact; "propose EvalSpec"
    gone, "adapts the open EEE schema" in.

- **Pass 15 (full-benchmark rerun incorporated):**
  - Author ran the full-vs-`limit=200` check (Qwen2.5-7B GSM8K plain/cot, hf backend,
    A100). Verified the numbers against the released grid: vLLM cell means are exactly
    plain 40.2 / cot 27.7 (temp-0, limit=200); hf full-vs-200 deltas ≤1.3 pp.
  - Added a compact paragraph to `app:exp-grid` (after the bootstrap CI) + a pointer in
    Limitations. Scoped strictly as: (a) **subset representativeness** confirmed
    (≤1.3 pp full vs 200) — answers the reviewer's limit=200 question; (b) **honest
    backend caveat** — hf vs vLLM differ and invert the plain/cot ordering, framed as
    *reinforcing* the thesis (backend = undocumented score-moving config), NOT a vLLM
    reproduction.
  - **Correctness checks:** the 14B CoT-collapse headline (58.5/2.5, vLLM, §5.1) is
    untouched and not contradicted — the new note is about the 7B and explicitly
    attributes the difference to the backend. Counting body unchanged at 4,844 words
    (addition is appendix/exempt). Balanced; no dangling refs.

- **Pass 16 (address reviewer round 2 — additions, no trimming per author):**
  - **#6 + #10 (missing runtime fields):** added `inference_backend`,
    `tokenizer_version`, `dataset_snapshot` (task commit hash / Croissant checksum)
    to the EvalSpec Recommended tier (body + appendix table), motivated by the
    backend rerun and Math-Verify versioning.
  - **#8 (no end-to-end record):** added Figure `fig:evalspec-json` — a full
    Required-tier conformant JSON record for the OLv2 Gemma-2-9B BBH case.
  - **#5 (temperature scope):** Limitations now states the <1% temperature result
    is GSM8K/greedy/parser-specific and backend-conditional; open-ended/safety tasks
    may differ.
  - **#2 (harness mislabel in release):** the Pipeline-attached-labels limitation now
    says the release marks inferred harness values (lm_eval-doc vs lighteval-runtime)
    so users don't propagate the doc label as truth.
  - Already-covered/minor (no change): #1 OLv2 dominance (ablation in place), #4
    single-extractor (in Limitations), #7 render artifacts (needs the compiled PDF),
    #9 interchange-effort survey (related work already tight).
  - Verified: balanced (figure 3/3, verbatim 1/1); Recommended rows aligned; no
    dangling refs. Additions are appendix/Limitations (page-exempt) + ~2 body lines.

- **Pass 17 (per author: backend/tokenizer/snapshot → future work):**
  - Moved `inference_backend`/`tokenizer_version`/`dataset_snapshot` OUT of the
    Recommended tier (removed table rows + body sentence + JSON lines); reframed as a
    Limitations "Extending the schema" future-work note (EvalSpec is an extensible
    EEE adaptation; these are the first community-expansion targets). Backend-rerun
    evidence retained in `app:exp-grid`.
  - Adoption payoff + migration/incentives already added (Adoption paragraph, Pass 16+).
  - Worked-example JSON (`fig:evalspec-json`) in place and now consistent (Required +
    prompt-hash only).
  - Verified: Recommended table 9 rows aligned; figure 3/3, verbatim 1/1; no dangling refs.

- **Pass 18 (compaction ~1000 words, lossless):** counting body **5,069 → 4,247
  (-822)**, fully lossless (every case-study number, citation, and `\ref` verified
  intact; all environments balanced). Compressed: EvalSpec section (merged
  field-selection + recommended-fields; worked-example → 1 sentence since table+JSON
  are in the appendix), §4.3 dominant-source + scoring-mode, Recommendations (3 role
  paragraphs → one), Misunderstanding Chain, factorial-grid intro, intro hook +
  framing, Related Work/EEE/Materials/audit prose, and the unified-figure caption.
  Body now ~6 pages. The remaining ~180 to a literal 1000 would start trimming
  secondary detail, so I stopped at the lossless boundary.

- **Pass 19 (remove all em-dashes):** all 64 prose `---` → commas; 4 em-dash pairs
  that wrapped internal comma-lists → parentheses (avoiding run-ons: intro "produced
  (harness, scoring mode, …) are"; SEA-LION "re-import (harness, …) were dropped";
  Misunderstanding-Chain "decisions (…) each known"; Conclusion "divergence (…) are
  largely absent"); 6 table-cell "no value" `---` → `--` (en-dash). `--` ranges
  (e.g.\ 55.10--59.60) untouched. Zero live `---` remain; no comma artifacts;
  balanced; no dangling refs.

## Loop status
Two consecutive sweeps (Pass 3 consistency + Pass 4 prose/structure) surfaced **no new
blocker/major** issues. The review loop has converged on everything resolvable without
(a) the author's judgment or (b) a LaTeX compile. Outstanding items are all in
**AUTHOR DECISION NEEDED** above; the only hard external dependency is an Overleaf build
to confirm a clean compile + the ≤8-page limit.
</content>
</invoke>
