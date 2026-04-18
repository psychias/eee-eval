# EEE Reproducibility Checklist

| Field | Difficulty | Coverage (%) | Implementation Note |
|-------|-----------|-------------|---------------------|
| eval_library.name | Low | 97.6 | Already logged by all dedicated leaderboards |
| eval_library.version | Low | 97.6 | One git tag or pip show away |
| generation_args.n_shot | Low | 96.7 | Already logged by most sources |
| generation_args.temperature | Medium | 0.0 | Requires per-run capture, not post-hoc |
| generation_args.prompt_template | High | 0.0 | Requires versioned template storage |
| additional_details.cot | Medium | 76.3 | Boolean flag, easy once pipeline supports it |
| scoring_mode | High | 0.0 | Not yet a schema field; requires definition |
| provenance_link | Medium | 0.0 | URL or DOI to upstream score source |

**Minimum viable report fields:** eval_library.name, eval_library.version, generation_args.n_shot, additional_details.cot
