# Changelog — acl_latex.tex

## Seventh-Pass: Section 2 Related Work — 2026-05-20

### What was written

Both subsections of §2 were converted from bullet-point placeholders to
flowing prose. The `%% DRAFT` comment blocks above each subsection were
preserved in place. The hypothetical table references (`Table: comparison of
documentation frameworks`, `Table: evaluation-context axes`) were dropped per
scope instructions (no new tables this pass).

A `\label{sec:related}` was added to `\section{Related Work}` to match the
cross-reference in the commented-out §1 draft.

---

#### §2.1 Evaluation Mispractices and Prior Audits (~300 words, 3 paragraphs)

**Paragraph 1 — Benchmark sensitivity.** Opens with the established finding
that scores vary with implementation choices, citing Alzahrani et al. (harness
choice, ~15 pp gap on MMLU), Biderman et al. (prompt format, tokenisation,
generation config), Singh et al. (Chatbot Arena audit, alias instability), and
Sainz et al. (contamination as a parallel reproducibility threat).

**Paragraph 2 — Differentiation.** States how this paper differs from prior
sensitivity studies: prior work shows *that* scores vary; this paper traces the
*concrete mechanisms* through which variation enters the public record as built.
Short paragraph (3 sentences) for rhythmic contrast.

**Paragraph 3 — Documentation frameworks.** Introduces Model Cards (Mitchell
et al. 2019) and Datasheets (Gebru et al. 2021) as the norm-setting prior
work, then positions three concurrent 2025 proposals (BenchmarkCards, EvalCards,
Eval Factsheets) as extending that norm to evaluation. States the distinction:
those frameworks are prescriptive; this paper is empirical first. Bridges to
EvalSpec v0.1 in Section 6.

---

#### §2.2 The EEE Schema (~190 words, 2 paragraphs)

**Paragraph 1 — Schema description.** Introduces EEE, states its purpose (one
model × one benchmark × one source per record), and describes all four blocks
with one sentence each, including the full field list for
`evaluation_results`.

**Paragraph 2 — Why a unified schema.** The "visible absence" argument: the
schema turns a missing field from a silent inference into a recordable null,
and it is the visibly-empty fields that motivate the paper. Closes with the
CC-BY-4.0 release statement.

---

### References used in §2

All nine references below were already present in `custom.bib`. No new BibTeX
entries were needed.

| Key | Paper | Status |
|-----|-------|--------|
| `alzahrani2024` | Alzahrani et al. 2024 — MMLU harness sensitivity | **[CITATION NEEDED]** stub; verify authors, title, venue |
| `biderman2024` | Biderman et al. 2024 — prompt/tokenisation sensitivity | **[CITATION NEEDED]** stub; verify authors, title, venue |
| `singh2025` | Singh et al. 2025 — Chatbot Arena audit | **[CITATION NEEDED]** stub; verify authors, title, venue |
| `sainz2023` | Sainz et al. 2023 — data contamination (EMNLP Findings) | **[CITATION NEEDED]** stub; verify full reference |
| `mitchell2019` | Mitchell et al. 2019 — Model Cards (FAccT) | **[CITATION NEEDED]** stub; verify full reference |
| `gebru2021` | Gebru et al. 2021 — Datasheets for Datasets (CACM) | **[CITATION NEEDED]** stub; verify full reference |
| `sokol2025` | Sokol et al. 2025 — BenchmarkCards | **[CITATION NEEDED]** stub; verify authors, title, venue |
| `dhar2025` | Dhar et al. 2025 — EvalCards | **[CITATION NEEDED]** stub; verify authors, title, venue |
| `bordes2025` | Bordes et al. 2025 — Eval Factsheets | **[CITATION NEEDED]** stub; verify authors, title, venue |

All nine stubs already carry `[CITATION NEEDED: verify ...]` markers in the
`note` field of `custom.bib`. They were not modified this pass; verification
against the actual papers is deferred to the citation-check pass.

---

### Scope confirmation

- §1, §3, §4, §5, §6, §7, abstract, appendix: **not touched**.
- `custom.bib`: **not modified** (all needed entries already present).
- Changes to `acl_latex.tex` are confined to lines 217–320 (the §2 block).

---

### Flags for other sections (do not edit — noted for future passes)

- §1 Introduction (line 198): contains placeholder text
  `(this needs to be rewritten, but im not that creative)` that should be
  removed before submission.
- §6 Discussion / EvalSpec subsection: has no `\label{}` yet; the §2.1
  bridge sentence uses the hardcoded `Section~6`. Add
  `\label{sec:evalspec}` to the subsection and update the cross-reference
  if the section number changes.
- Several citation stubs throughout the paper carry `[CITATION NEEDED]`
  markers; a full citation-check pass is recommended before submission.
