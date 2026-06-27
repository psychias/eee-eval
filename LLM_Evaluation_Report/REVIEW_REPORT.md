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

## Loop status
Two consecutive sweeps (Pass 3 consistency + Pass 4 prose/structure) surfaced **no new
blocker/major** issues. The review loop has converged on everything resolvable without
(a) the author's judgment or (b) a LaTeX compile. Outstanding items are all in
**AUTHOR DECISION NEEDED** above; the only hard external dependency is an Overleaf build
to confirm a clean compile + the ≤8-page limit.
</content>
</invoke>
