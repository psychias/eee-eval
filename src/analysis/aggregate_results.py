"""
aggregate_results.py — flatten all per-benchmark JSONs into one master CSV.

Field paths are matched exactly to the JSON structure produced by
PaperConverter._convert_model() → EvaluationLog.model_dump(mode='json').

JSON structure written by PaperWriter (one EvaluationResult per file):
  {
    "schema_version": "...",
    "evaluation_id":  "...",
    "retrieved_timestamp": "...",
    "source_metadata": {
        "source_name": "arXiv:...",
        "source_type": "documentation",
        "source_organization_name": "arXiv",
        "source_organization_url": "https://...",
        "evaluator_relationship": "first_party|third_party"
    },
    "eval_library": { "name": "lm_eval|internal|...", "version": "unknown" },
    "model_info": {
        "name": "Llama-3-70B",
        "id":   "meta-llama/Llama-3-70B",
        "developer": "meta-llama",
        "parameter_count": "70B",        # optional
        "model_alignment": "instruct",   # optional
        "quantization": "BF16"           # optional
    },
    "evaluation_results": [              # always exactly ONE item per file
      {
        "evaluation_name": "MMLU",
        "source_data": { ... },
        "metric_config": {
            "metric_name": "accuracy",
            "lower_is_better": false,
            "score_type": "continuous",
            "min_score": 0.0,
            "max_score": 1.0
        },
        "score_details": {
            "score": 0.8732,
            "details": { ... }           # optional — relative_to, score_type
        },
        "generation_config": {
            "generation_args": {         # optional — omitted if all None
                "shots": 5,
                "temperature": 0.0,
                "top_p": null,
                "chain_of_thought": null,
                "prompt_template": null
            },
            "additional_details": {      # always present — dict[str, str]
                "source": "arXiv:...",
                "extraction_confidence": "high|medium|low|llm|prose",
                "harness": "lm_eval|internal|...",
                "scoring_method": "log_likelihood|generation",  # optional
                "protocol_source": "paper_default|explicit|...",  # optional
                "page_number": "3",      # optional
                "table": "Table 2",      # optional
                "table_caption": "..."   # optional
            }
        }
      }
    ]
  }
"""

import csv
import json
import pathlib
import collections

ROOT    = pathlib.Path(__file__).resolve().parent.parent.parent
OUT_DIR = ROOT / "data" / "aggregated"
OUT_DIR.mkdir(parents=True, exist_ok=True)

rows: list[dict] = []
# coverage[benchmark][field] = count of non-empty values
coverage: dict = collections.defaultdict(lambda: collections.defaultdict(int))

for f in (ROOT / "data").rglob("*.json"):
    if "aggregated" in str(f):
        continue
    try:
        rec = json.loads(f.read_text(encoding="utf-8"))

        # ── Top-level record fields ─────────────────────────────────────────
        mi  = rec.get("model_info", {})
        src = rec.get("source_metadata", {})
        lib = rec.get("eval_library", {})

        # ── Each file has exactly ONE evaluation_result ─────────────────────
        for er in rec.get("evaluation_results", []):
            mc = er.get("metric_config", {})
            sd = er.get("score_details", {})
            gc = er.get("generation_config", {})

            # generation_args is optional (excluded when all fields are None)
            ga = gc.get("generation_args") or {}

            # additional_details is always present — dict[str, str]
            ad = gc.get("additional_details") or {}

            # score_details.details holds relative_to / comparability_warning
            sd_details = sd.get("details") or {}

            row = {
                # ── Model identity ──────────────────────────────────────────
                "model":                  mi.get("name", ""),
                "model_id":               mi.get("id", ""),
                "developer":              mi.get("developer", ""),
                "parameter_count":        mi.get("parameter_count", ""),
                "model_alignment":        mi.get("model_alignment", ""),
                "quantization":           mi.get("quantization", ""),

                # ── Benchmark + score ───────────────────────────────────────
                "benchmark":              er.get("evaluation_name", ""),
                "score":                  sd.get("score", ""),
                "lower_is_better":        mc.get("lower_is_better", ""),
                "min_score":              mc.get("min_score", ""),
                "max_score":              mc.get("max_score", ""),
                "metric_name":            mc.get("metric_name", ""),
                "score_type":             sd_details.get("score_type", "absolute"),
                "relative_to":            sd_details.get("relative_to", ""),

                # ── Evaluation protocol (generation_args) ───────────────────
                # Paths: generation_config.generation_args.*
                "shots":                  ga.get("shots", ""),
                "temperature":            ga.get("temperature", ""),
                "top_p":                  ga.get("top_p", ""),
                "reasoning":              ga.get("reasoning", ""),
                "chain_of_thought":       ga.get("chain_of_thought", ""),
                "prompt_template":        ga.get("prompt_template", ""),

                # ── Extraction provenance (additional_details) ───────────────
                # Paths: generation_config.additional_details.*
                "extraction_confidence":  ad.get("extraction_confidence", ""),
                # harness: prefer additional_details.harness, fall back to eval_library.name
                "harness":                ad.get("harness", "") or lib.get("name", ""),
                "scoring_method":         ad.get("scoring_method", ""),
                "protocol_source":        ad.get("protocol_source", ""),
                "page_number":            ad.get("page_number", ""),
                "table_id":               ad.get("table", ""),
                "table_caption":          ad.get("table_caption", ""),

                # ── Source metadata ─────────────────────────────────────────
                # source_name may be a paper title; arxiv_id is always
                # the arXiv identifier for consistent grouping.
                "source":                 ad.get("source", src.get("source_name", "")),
                "source_name":            src.get("source_name", ""),
                # evaluator_relationship: EvaluatorRelationship enum → string
                "evaluator_relationship": src.get("evaluator_relationship", ""),
                "eval_library":           lib.get("name", ""),
                "eval_library_version":   lib.get("version", ""),

                # ── Record provenance ───────────────────────────────────────
                "evaluation_id":          rec.get("evaluation_id", ""),
                "retrieved_timestamp":    rec.get("retrieved_timestamp", ""),
                "file":                   str(f.relative_to(ROOT)),
            }
            rows.append(row)

            # Coverage tracking — which metadata fields are documented
            b = row["benchmark"]
            for field in ("shots", "temperature", "top_p",
                          "prompt_template", "harness", "chain_of_thought",
                          "reasoning"):
                val = row.get(field, "")
                if val not in ("", None, "unknown"):
                    coverage[b][field] += 1
            coverage[b]["total"] += 1

    except Exception as e:
        print(f"  skip {f.name}: {e}")

if rows:
    keys = list(rows[0].keys())
    with open(OUT_DIR / "all_results.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    with open(OUT_DIR / "coverage_stats.json", "w", encoding="utf-8") as fh:
        json.dump(dict(coverage), fh, indent=2)

    # Also collect the native coverage_report_*.json files the pipeline writes
    native_reports = sorted(
        (ROOT / "scripts" / "scrapers" / "raw").glob("coverage_report_*.json")
    )
    if native_reports:
        latest = json.loads(native_reports[-1].read_text(encoding="utf-8"))
        with open(OUT_DIR / "pipeline_coverage_report.json", "w", encoding="utf-8") as fh:
            json.dump(latest, fh, indent=2)
        print(f"  pipeline coverage report -> data/aggregated/pipeline_coverage_report.json")

    models      = len({r["model"]     for r in rows})
    benchmarks  = len({r["benchmark"] for r in rows})
    sources     = len({r["source"]    for r in rows})
    developers  = len({r["developer"] for r in rows})

    print(f"OK {len(rows)} records -> data/aggregated/all_results.csv")
    print(f"  {models} models  |  {benchmarks} benchmarks  |  "
          f"{sources} sources  |  {developers} developers")

    # Quick metadata coverage summary
    print("\n  Metadata documentation rates:")
    for field in ("shots", "temperature", "prompt_template", "harness", "chain_of_thought", "reasoning"):
        filled = sum(1 for r in rows if r.get(field) not in ("", None, "unknown"))
        pct = filled / len(rows) * 100
        print(f"    {field:20s}  {filled:4d}/{len(rows)}  ({pct:.1f}%)")
else:
    print("  no records found — run extraction first (pipeline: 1b — extract batch)")
