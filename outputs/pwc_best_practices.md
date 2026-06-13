# PWC Data Ingestion Best Practices

## Summary of Artefact Types

| Artefact Type | Count | Example |
|---------------|-------|---------|
| Score > 1000 (possible param count) | 0 | Score field contains parameter counts instead of benchmark scores |
| Param pattern in score (e.g., "7B") | 0 | Score string matches model-size notation |
| Variant-tag leakage in prompt_template | 2 | Non-template text (cot, chat, instruct) in template field |
| Duplicate (model, benchmark, score) | 0 | Exact triple duplicates from multiple submissions |
| **Total unique artefact records** | **2** | **0.3% of 576 PWC records** |

## Three Detection Rules

1. **Numeric range check**: Flag any record where `score > 1000` or `score` matches
   the pattern `\d+[Bb]` (e.g., "7B", "70B"). These are likely parameter counts
   or model-size descriptors that were scraped into the score field.

2. **Template field validation**: Reject records where `prompt_template` contains
   variant identifiers (`cot`, `chat`, `instruct`, `-v`) rather than actual
   template text. This indicates the field was used to tag model variants rather
   than store generation configuration.

3. **Deduplication**: Remove exact `(model_id, benchmark, score)` triples. Multiple
   identical entries arise from scraping the same result from different paper tables
   or re-submissions.

## Recommended Ingestion Checklist

- [ ] Parse numeric scores and reject values outside the benchmark's documented range
      (`min_score`, `max_score` from schema)
- [ ] Strip whitespace and normalise model IDs before deduplication
- [ ] Validate `prompt_template` is either null/empty or contains template markup
      (e.g., Jinja2 syntax, placeholder tokens)
- [ ] Cross-reference `benchmark` names against a canonical list to merge aliases
- [ ] Log `extraction_confidence` and filter at a threshold (e.g., >= 0.8) for
      automated pipelines
- [ ] After cleaning, re-check per-benchmark record counts to ensure no benchmark
      lost all its data

## Per-Benchmark Artefact Rate

| benchmark        |   total |   artefacts |   rate_% |
|:-----------------|--------:|------------:|---------:|
| MBPP             |      80 |           1 | 1.25     |
| GSM8K            |     111 |           1 | 0.900901 |
| DROP             |       1 |           0 | 0        |
| ARC-Challenge    |      35 |           0 | 0        |
| GPQA             |       4 |           0 | 0        |
| HellaSwag        |      82 |           0 | 0        |
| HumanEval        |       7 |           0 | 0        |
| BBH              |       1 |           0 | 0        |
| IFEval           |       4 |           0 | 0        |
| MATH             |     100 |           0 | 0        |
| MATH-500         |       1 |           0 | 0        |
| MMLU             |       1 |           0 | 0        |
| MMLU-Pro         |       1 |           0 | 0        |
| NaturalQuestions |      50 |           0 | 0        |
| TruthfulQA       |      29 |           0 | 0        |
| WinoGrande       |      69 |           0 | 0        |
