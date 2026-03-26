"""
Comprehensive data quality audit for all EEE JSON records.

Checks:
  1. Schema validation (jsonschema)
  2. Shots field audit: 0 vs null vs absent — are zeros genuine?
  3. Harness version coverage — is "unknown" acceptable?
  4. Score sanity: ranges, outliers, negative scores, parameter-leak detection
  5. Metadata coverage: per-source breakdown
  6. Within-source duplicate / conflict detection
  7. Model identity consistency
  8. Benchmark naming consistency
  9. Required-field completeness
"""

import json, os, sys, glob, re
from collections import defaultdict, Counter
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SCHEMA_PATH = Path(__file__).resolve().parent.parent / "eval.schema.json"

# Known ground-truth shots per (source, benchmark)
# These are the values hardcoded in add_leaderboard_records.py
KNOWN_SHOTS = {
    # Open LLM Leaderboard v2
    ("open_llm_leaderboard_v2", "IFEval"):      0,
    ("open_llm_leaderboard_v2", "BBH"):          3,
    ("open_llm_leaderboard_v2", "MATH-500"):     4,
    ("open_llm_leaderboard_v2", "GPQA"):         0,
    ("open_llm_leaderboard_v2", "MuSR"):         0,
    ("open_llm_leaderboard_v2", "MMLU-Pro"):     5,
    # AlpacaEval
    ("alpacaeval2", "AlpacaEval 2.0"):           0,
    ("alpacaeval2", "AlpacaEval 2.0 LC"):        0,
    # BFCL
    ("bfcl", "BFCL-v3"):                         0,
    ("bfcl", "BFCL-v3 AST"):                     0,
    ("bfcl", "BFCL-v3 Live"):                    0,
    # BigCodeBench
    ("bigcodebench", "BigCodeBench-Complete"):    0,
    ("bigcodebench", "BigCodeBench-Instruct"):    0,
    # EvalPlus
    ("evalplus", "HumanEval+"):                   0,
    ("evalplus", "MBPP+"):                        3,
    # SWE-Bench
    ("swe_bench", "SWE-Bench-Verified"):          0,
}

# Benchmarks where shots is N/A (preference / arena based)
SHOTS_NA_BENCHMARKS = {
    "Arena Elo", "MT-Bench", "WildBench v2", "WildBench v2 (adj)"
}


def load_all_records():
    """Load all JSON files from data/ subfolders (excluding aggregated)."""
    records = []
    source_dirs = [d for d in DATA_DIR.iterdir()
                   if d.is_dir() and d.name not in ("aggregated",)]
    for src_dir in sorted(source_dirs):
        source_name = src_dir.name
        for json_path in src_dir.rglob("*.json"):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                records.append((source_name, json_path, data))
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                records.append((source_name, json_path, {"__parse_error__": str(e)}))
    return records


def audit_schema_validation(records):
    """Check all records against the EEE schema."""
    issues = []
    try:
        import jsonschema
        with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
            schema = json.load(f)
    except ImportError:
        return [("SKIP", "N/A", "jsonschema not installed — skipping schema validation")]
    except FileNotFoundError:
        return [("SKIP", "N/A", f"Schema file not found: {SCHEMA_PATH}")]

    for source, path, data in records:
        if "__parse_error__" in data:
            issues.append(("CRITICAL", str(path), f"JSON parse error: {data['__parse_error__']}"))
            continue
        try:
            jsonschema.validate(data, schema)
        except jsonschema.ValidationError as e:
            issues.append(("ERROR", str(path), f"Schema validation failed: {e.message[:200]}"))
    return issues


def audit_shots(records):
    """Audit the shots field across all records."""
    issues = []
    stats = defaultdict(lambda: Counter())  # source -> Counter of shot values

    for source, path, data in records:
        if "__parse_error__" in data:
            continue
        for er in data.get("evaluation_results", []):
            bench = er.get("evaluation_name", "?")
            gen_args = er.get("generation_config", {}).get("generation_args", {})
            shots_val = gen_args.get("shots")  # could be 0, int, None, or missing

            has_shots_key = "shots" in gen_args
            source_bench_key = (source, bench)

            # Track distribution
            if has_shots_key:
                stats[source][f"shots={shots_val}"] += 1
            else:
                stats[source]["shots=ABSENT"] += 1

            # Check against ground-truth
            if source_bench_key in KNOWN_SHOTS:
                expected = KNOWN_SHOTS[source_bench_key]
                if not has_shots_key:
                    issues.append(("WARNING", str(path),
                        f"[{source}/{bench}] shots field ABSENT but expected {expected}"))
                elif shots_val != expected:
                    issues.append(("ERROR", str(path),
                        f"[{source}/{bench}] shots={shots_val} but expected {expected}"))

            # Flag zeros on benchmarks where we're uncertain
            if has_shots_key and shots_val == 0 and source_bench_key not in KNOWN_SHOTS \
                    and bench not in SHOTS_NA_BENCHMARKS:
                issues.append(("INFO", str(path),
                    f"[{source}/{bench}] shots=0 — verify this is genuinely 0-shot, not a default"))

            # Flag shots on NA benchmarks
            if has_shots_key and shots_val is not None and bench in SHOTS_NA_BENCHMARKS:
                issues.append(("WARNING", str(path),
                    f"[{source}/{bench}] shots={shots_val} but this benchmark is preference-based (shots N/A)"))

    return issues, stats


def audit_harness_version(records):
    """Audit eval_library name and version fields."""
    issues = []
    stats = defaultdict(lambda: Counter())  # source -> Counter of (harness, version)

    for source, path, data in records:
        if "__parse_error__" in data:
            continue
        lib = data.get("eval_library", {})
        name = lib.get("name", "MISSING")
        version = lib.get("version", "MISSING")

        stats[source][(name, version)] += 1

        if name == "MISSING":
            issues.append(("ERROR", str(path),
                f"[{source}] eval_library.name is MISSING (required field)"))
        if version == "unknown":
            issues.append(("INFO", str(path),
                f"[{source}] eval_library.version='unknown' — harness: {name}"))
        elif version == "MISSING":
            issues.append(("ERROR", str(path),
                f"[{source}] eval_library.version is MISSING (required field)"))

    return issues, stats


def audit_scores(records):
    """Audit score values for sanity."""
    issues = []
    score_ranges = defaultdict(list)  # (source, bench) -> [scores]

    for source, path, data in records:
        if "__parse_error__" in data:
            continue
        for er in data.get("evaluation_results", []):
            bench = er.get("evaluation_name", "?")
            score_details = er.get("score_details", {})
            score = score_details.get("score")

            if score is None:
                issues.append(("ERROR", str(path),
                    f"[{source}/{bench}] score is None/missing"))
                continue

            score_ranges[(source, bench)].append(score)
            metric_cfg = er.get("metric_config", {})
            min_s = metric_cfg.get("min_score")
            max_s = metric_cfg.get("max_score")

            # Check if score is outside declared range
            if min_s is not None and score < min_s:
                issues.append(("ERROR", str(path),
                    f"[{source}/{bench}] score={score} < min_score={min_s}"))
            if max_s is not None and score > max_s:
                issues.append(("ERROR", str(path),
                    f"[{source}/{bench}] score={score} > max_score={max_s}"))

            # Detect parameter-count leaks (very large scores on 0-100 benchmarks)
            if max_s is not None and max_s <= 100 and score > 100:
                issues.append(("CRITICAL", str(path),
                    f"[{source}/{bench}] score={score} on 0-100 scale — likely parameter count leak"))

            # Detect suspicious zeros
            if score == 0.0:
                issues.append(("WARNING", str(path),
                    f"[{source}/{bench}] score=0.0 — may be a missing/default value"))

            # Negative score on non-negative benchmark
            if min_s is not None and min_s >= 0 and score < 0:
                issues.append(("ERROR", str(path),
                    f"[{source}/{bench}] score={score} negative on non-negative benchmark"))

    return issues, score_ranges


def audit_metadata_coverage(records):
    """Compute metadata field coverage per source."""
    fields = ["shots", "temperature", "top_p", "prompt_template", "chain_of_thought"]
    coverage = defaultdict(lambda: {f: {"filled": 0, "total": 0} for f in fields})
    harness_coverage = defaultdict(lambda: {"known": 0, "unknown": 0, "total": 0})

    for source, path, data in records:
        if "__parse_error__" in data:
            continue
        lib = data.get("eval_library", {})
        lib_name = lib.get("name", "")
        harness_coverage[source]["total"] += 1
        if lib_name and lib_name != "unknown":
            harness_coverage[source]["known"] += 1
        else:
            harness_coverage[source]["unknown"] += 1

        for er in data.get("evaluation_results", []):
            gen_args = er.get("generation_config", {}).get("generation_args", {})
            for f in fields:
                coverage[source][f]["total"] += 1
                val = gen_args.get(f)
                if val is not None and val != "" and val != "unknown":
                    coverage[source][f]["filled"] += 1

    return coverage, harness_coverage


def audit_duplicates(records):
    """Detect within-source duplicate (model, benchmark) pairs."""
    issues = []
    # (source, model_id, benchmark) -> [(score, path)]
    seen = defaultdict(list)

    for source, path, data in records:
        if "__parse_error__" in data:
            continue
        model_id = data.get("model_info", {}).get("id", "?")
        for er in data.get("evaluation_results", []):
            bench = er.get("evaluation_name", "?")
            score = er.get("score_details", {}).get("score")
            key = (source, model_id, bench)
            seen[key].append((score, str(path)))

    for key, entries in seen.items():
        if len(entries) > 1:
            scores = [e[0] for e in entries]
            paths = [e[1] for e in entries]
            if len(set(scores)) > 1:
                issues.append(("WARNING", paths[0],
                    f"[{key[0]}/{key[2]}] model={key[1]} has {len(entries)} records with different scores: {scores}"))
            else:
                issues.append(("INFO", paths[0],
                    f"[{key[0]}/{key[2]}] model={key[1]} has {len(entries)} duplicate records (same score)"))

    return issues


def audit_model_identity(records):
    """Check model identity consistency."""
    issues = []
    for source, path, data in records:
        if "__parse_error__" in data:
            continue
        mi = data.get("model_info", {})
        name = mi.get("name", "")
        mid = mi.get("id", "")
        dev = mi.get("developer", "")

        if not name:
            issues.append(("ERROR", str(path), f"[{source}] model_info.name is empty"))
        if not mid:
            issues.append(("ERROR", str(path), f"[{source}] model_info.id is empty"))
        if not dev:
            issues.append(("WARNING", str(path), f"[{source}] model_info.developer is empty for {mid}"))

        # Check id format (should be developer/model_name)
        if mid and "/" not in mid:
            issues.append(("WARNING", str(path),
                f"[{source}] model_info.id='{mid}' doesn't follow developer/model format"))

    return issues


def audit_benchmark_naming(records):
    """Check benchmark naming consistency across sources."""
    bench_by_source = defaultdict(set)
    for source, path, data in records:
        if "__parse_error__" in data:
            continue
        for er in data.get("evaluation_results", []):
            bench = er.get("evaluation_name", "?")
            bench_by_source[source].add(bench)

    # Detect near-duplicates across sources
    all_benchmarks = set()
    for benchmarks in bench_by_source.values():
        all_benchmarks.update(benchmarks)

    issues = []
    bench_list = sorted(all_benchmarks)
    for i, b1 in enumerate(bench_list):
        for b2 in bench_list[i+1:]:
            if b1.lower().replace("-", "").replace("_", "").replace(" ", "") == \
               b2.lower().replace("-", "").replace("_", "").replace(" ", ""):
                if b1 != b2:
                    issues.append(("WARNING", "N/A",
                        f"Near-duplicate benchmark names: '{b1}' vs '{b2}'"))

    return issues, bench_by_source


def audit_required_fields(records):
    """Check all schema-required fields are present and non-empty."""
    issues = []
    required_top = ["schema_version", "evaluation_id", "retrieved_timestamp",
                    "source_metadata", "model_info", "eval_library", "evaluation_results"]

    for source, path, data in records:
        if "__parse_error__" in data:
            continue
        for field in required_top:
            if field not in data:
                issues.append(("ERROR", str(path), f"[{source}] Missing required field: {field}"))
            elif data[field] is None:
                issues.append(("ERROR", str(path), f"[{source}] Required field is null: {field}"))

        # Check source_metadata required fields
        sm = data.get("source_metadata", {})
        for sf in ["source_type", "source_organization_name", "evaluator_relationship"]:
            if sf not in sm or not sm[sf]:
                issues.append(("ERROR", str(path), f"[{source}] Missing source_metadata.{sf}"))

        # Check eval_library
        el = data.get("eval_library", {})
        for ef in ["name", "version"]:
            if ef not in el:
                issues.append(("ERROR", str(path), f"[{source}] Missing eval_library.{ef}"))

        # Check evaluation_results non-empty
        ers = data.get("evaluation_results", [])
        if not ers:
            issues.append(("ERROR", str(path), f"[{source}] evaluation_results is empty"))

        for er in ers:
            for rf in ["evaluation_name", "source_data", "metric_config", "score_details"]:
                if rf not in er:
                    issues.append(("ERROR", str(path),
                        f"[{source}] Missing evaluation_results[].{rf}"))

    return issues


def print_section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def main():
    print("Loading all records...")
    records = load_all_records()
    total_files = len(records)
    total_eval_results = sum(
        len(d.get("evaluation_results", []))
        for _, _, d in records if "__parse_error__" not in d
    )
    parse_errors = sum(1 for _, _, d in records if "__parse_error__" in d)

    # Count per source
    source_counts = Counter()
    source_eval_counts = Counter()
    for source, _, data in records:
        source_counts[source] += 1
        if "__parse_error__" not in data:
            source_eval_counts[source] += len(data.get("evaluation_results", []))

    print(f"\nTotal files:             {total_files}")
    print(f"Parse errors:            {parse_errors}")
    print(f"Total evaluation records: {total_eval_results}")
    print(f"\nPer-source file counts:")
    for src in sorted(source_counts):
        print(f"  {src:30s} {source_counts[src]:6d} files  {source_eval_counts[src]:6d} eval records")

    all_issues = []

    # === 1. Schema Validation ===
    print_section("1. SCHEMA VALIDATION")
    schema_issues = audit_schema_validation(records)
    all_issues.extend(schema_issues)
    error_count = sum(1 for sev, _, _ in schema_issues if sev in ("ERROR", "CRITICAL"))
    print(f"  Schema errors: {error_count}")
    for sev, path, msg in schema_issues[:20]:
        if sev in ("ERROR", "CRITICAL"):
            print(f"    [{sev}] {msg}")
    if error_count > 20:
        print(f"    ... and {error_count - 20} more")

    # === 2. Shots Audit ===
    print_section("2. SHOTS FIELD AUDIT")
    shots_issues, shots_stats = audit_shots(records)
    all_issues.extend(shots_issues)
    print("\n  Shots distribution per source:")
    for src in sorted(shots_stats):
        print(f"    {src}:")
        for val, cnt in shots_stats[src].most_common():
            print(f"      {val:20s} {cnt:6d}")

    errors = [i for i in shots_issues if i[0] in ("ERROR", "CRITICAL")]
    warnings = [i for i in shots_issues if i[0] == "WARNING"]
    print(f"\n  Shots errors: {len(errors)}")
    for sev, path, msg in errors[:10]:
        print(f"    [{sev}] {msg}")
    print(f"  Shots warnings: {len(warnings)}")
    for sev, path, msg in warnings[:10]:
        print(f"    [{sev}] {msg}")

    # === 3. Harness Version Audit ===
    print_section("3. HARNESS / EVAL_LIBRARY AUDIT")
    version_issues, version_stats = audit_harness_version(records)

    print("\n  Harness identity per source:")
    for src in sorted(version_stats):
        print(f"    {src}:")
        for (name, ver), cnt in version_stats[src].most_common():
            print(f"      {name:20s} version={ver:15s}  ({cnt} files)")

    unknown_total = sum(v["unknown"] for v in
        {s: {"unknown": sum(1 for (n,_), c in version_stats[s].items()
             if n == "unknown" for _ in range(c)), "total": 0}
         for s in version_stats}.values())

    version_unknown = sum(1 for sev, _, msg in version_issues if "version='unknown'" in msg)
    print(f"\n  Files with harness version='unknown': {version_unknown}")
    print(f"  (This means the specific harness version number was not available — ")
    print(f"   it does NOT mean the harness identity is unknown)")

    # === 4. Score Sanity ===
    print_section("4. SCORE SANITY CHECKS")
    score_issues, score_ranges = audit_scores(records)
    all_issues.extend(score_issues)

    critical = [i for i in score_issues if i[0] == "CRITICAL"]
    errors = [i for i in score_issues if i[0] == "ERROR"]
    warnings = [i for i in score_issues if i[0] == "WARNING"]
    print(f"  Critical (param leaks): {len(critical)}")
    for sev, path, msg in critical[:10]:
        print(f"    [{sev}] {msg}")
    print(f"  Errors (out of range):  {len(errors)}")
    for sev, path, msg in errors[:10]:
        print(f"    [{sev}] {msg}")
    print(f"  Warnings (zero scores): {len(warnings)}")
    for sev, _, msg in warnings[:10]:
        print(f"    [{sev}] {msg}")

    # Print score range summary
    print("\n  Score ranges per (source, benchmark) [top 15 by record count]:")
    sorted_ranges = sorted(score_ranges.items(), key=lambda x: -len(x[1]))[:15]
    for (src, bench), scores in sorted_ranges:
        print(f"    {src}/{bench}: min={min(scores):.2f} max={max(scores):.2f} "
              f"mean={sum(scores)/len(scores):.2f} n={len(scores)}")

    # === 5. Metadata Coverage ===
    print_section("5. METADATA COVERAGE")
    coverage, harness_cov = audit_metadata_coverage(records)

    header = f"  {'Source':30s} {'harness':>8s} {'shots':>8s} {'temp':>8s} {'top_p':>8s} {'prompt':>8s} {'CoT':>8s}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for src in sorted(coverage):
        hc = harness_cov[src]
        h_pct = f"{hc['known']/max(hc['total'],1)*100:.0f}%"
        row = f"  {src:30s} {h_pct:>8s}"
        for f in ["shots", "temperature", "top_p", "prompt_template", "chain_of_thought"]:
            c = coverage[src][f]
            pct = c["filled"] / max(c["total"], 1) * 100
            row += f" {pct:7.1f}%"
        print(row)

    # Overall
    print()
    totals = {f: {"filled": 0, "total": 0} for f in ["shots", "temperature", "top_p", "prompt_template", "chain_of_thought"]}
    for src in coverage:
        for f in totals:
            totals[f]["filled"] += coverage[src][f]["filled"]
            totals[f]["total"] += coverage[src][f]["total"]
    h_total = sum(v["total"] for v in harness_cov.values())
    h_known = sum(v["known"] for v in harness_cov.values())
    row = f"  {'OVERALL':30s} {h_known/max(h_total,1)*100:.0f}%".ljust(40)
    for f in totals:
        pct = totals[f]["filled"] / max(totals[f]["total"], 1) * 100
        row += f" {pct:7.1f}%"
    print(row)

    # === 6. Duplicates ===
    print_section("6. WITHIN-SOURCE DUPLICATES")
    dup_issues = audit_duplicates(records)
    all_issues.extend(dup_issues)
    conflicts = [i for i in dup_issues if i[0] == "WARNING"]
    dupes = [i for i in dup_issues if i[0] == "INFO"]
    print(f"  Score conflicts (same model+bench, different scores): {len(conflicts)}")
    for sev, _, msg in conflicts[:15]:
        print(f"    {msg}")
    print(f"  Exact duplicates (same model+bench, same score):     {len(dupes)}")

    # === 7. Model Identity ===
    print_section("7. MODEL IDENTITY CHECKS")
    model_issues = audit_model_identity(records)
    all_issues.extend(model_issues)
    errors = [i for i in model_issues if i[0] == "ERROR"]
    warnings = [i for i in model_issues if i[0] == "WARNING"]
    print(f"  Errors:   {len(errors)}")
    for _, _, msg in errors[:10]:
        print(f"    {msg}")
    print(f"  Warnings: {len(warnings)}")
    for _, _, msg in warnings[:10]:
        print(f"    {msg}")

    # === 8. Benchmark Naming ===
    print_section("8. BENCHMARK NAMING CONSISTENCY")
    bench_issues, bench_by_source = audit_benchmark_naming(records)
    all_issues.extend(bench_issues)
    print(f"  Near-duplicate benchmark names detected: {len(bench_issues)}")
    for _, _, msg in bench_issues[:20]:
        print(f"    {msg}")
    print(f"\n  Total unique benchmarks: {sum(len(v) for v in bench_by_source.values())}")
    for src in sorted(bench_by_source):
        benchmarks = sorted(bench_by_source[src])
        print(f"    {src}: {benchmarks}")

    # === 9. Required Fields ===
    print_section("9. REQUIRED FIELD COMPLETENESS")
    req_issues = audit_required_fields(records)
    all_issues.extend(req_issues)
    print(f"  Missing required fields: {len(req_issues)}")
    for _, _, msg in req_issues[:20]:
        print(f"    {msg}")

    # === SUMMARY ===
    print_section("QUALITY AUDIT SUMMARY")
    severity_counts = Counter(sev for sev, _, _ in all_issues)
    print(f"  Total files scanned:       {total_files}")
    print(f"  Total eval records:        {total_eval_results}")
    print(f"  CRITICAL issues:           {severity_counts.get('CRITICAL', 0)}")
    print(f"  ERROR issues:              {severity_counts.get('ERROR', 0)}")
    print(f"  WARNING issues:            {severity_counts.get('WARNING', 0)}")
    print(f"  INFO (notes):              {severity_counts.get('INFO', 0)}")

    if severity_counts.get("CRITICAL", 0) > 0 or severity_counts.get("ERROR", 0) > 0:
        print(f"\n  *** DATA HAS ISSUES REQUIRING ATTENTION ***")
        return 1
    else:
        print(f"\n  Data quality: PASS (no critical or error-level issues)")
        return 0


if __name__ == "__main__":
    sys.exit(main())
