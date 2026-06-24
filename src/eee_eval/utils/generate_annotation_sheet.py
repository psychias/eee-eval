"""
Generate a structured CSV annotation sheet for human reviewers.

For each extracted record, the annotator classifies:
  1 = Correct
  2 = Incorrect
  3 = No detail provided (paper doesn't give enough info to verify)

Output: data/arxiv_extraction_general/samples/human_annotation_sheet.csv
"""

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SAMPLES_DIR = ROOT / "data" / "arxiv_extraction_general" / "samples"
OUTPUT_CSV = SAMPLES_DIR / "human_annotation_sheet.csv"

COLUMNS = [
    "row_id",
    "sample_set",
    "paper_folder",
    "arxiv_id",
    "paper_title",
    "model_name",
    "benchmark",
    "metric",
    "extracted_score",
    "normalized_score",
    "shots",
    "eval_library",
    "is_baseline",
    "json_file",
    # --- human fills these ---
    "annotation",          # 1=Correct, 2=Incorrect, 3=No detail provided
    "annotator_notes",
]


def extract_row(json_path: Path, sample_set: str, paper_folder: str) -> dict | None:
    """Parse one JSON evaluation file into an annotation row."""
    try:
        data = json.loads(json_path.read_bytes())
    except (json.JSONDecodeError, OSError):
        return None

    source = data.get("source_metadata", {})
    details = source.get("additional_details", {})
    model = data.get("model_info", {})
    lib = data.get("eval_library", {})

    results = data.get("evaluation_results", [])
    if not results:
        return None

    r = results[0]
    score = None
    norm_score = None

    # Try multiple score locations
    if "results" in r and r["results"]:
        score = r["results"][0].get("score")
        norm_score = r["results"][0].get("normalized_score")
    elif "score_details" in r:
        score = r["score_details"].get("score")
        norm_score = r["score_details"].get("normalized_score")

    metric_cfg = r.get("metric_config", {})
    gen_cfg = r.get("generation_config", {}).get("generation_args", {})
    eval_spec = r.get("eval_spec", {})

    shots = eval_spec.get("shots") or gen_cfg.get("n_shot")

    return {
        "sample_set": sample_set,
        "paper_folder": paper_folder,
        "arxiv_id": details.get("arxiv_id", ""),
        "paper_title": details.get("paper_title", ""),
        "model_name": model.get("name", ""),
        "benchmark": r.get("evaluation_name", ""),
        "metric": metric_cfg.get("metric_name", ""),
        "extracted_score": score,
        "normalized_score": norm_score,
        "shots": shots,
        "eval_library": lib.get("name", ""),
        "is_baseline": details.get("is_baseline", ""),
        "json_file": str(json_path.relative_to(SAMPLES_DIR)),
        "annotation": "",
        "annotator_notes": "",
    }


def collect_entries(folder: Path, sample_set: str) -> list[dict]:
    """Collect all JSON entries from a sample folder."""
    rows = []
    if not folder.exists():
        return rows

    for json_path in sorted(folder.rglob("*.json")):
        # paper_folder is the first directory level under the sample set folder
        rel = json_path.relative_to(folder)
        paper_folder = rel.parts[0] if rel.parts else ""

        row = extract_row(json_path, sample_set, paper_folder)
        if row:
            rows.append(row)
    return rows


def main():
    all_rows = []

    # Score verification sample
    sv_dir = SAMPLES_DIR / "score_verification_sample"
    all_rows.extend(collect_entries(sv_dir, "score_verification"))

    # Eval harness - WITH
    eh_with_dir = SAMPLES_DIR / "eval_harness_sample" / "with_evaluation_harmness"
    all_rows.extend(collect_entries(eh_with_dir, "eval_harness_WITH"))

    # Eval harness - WITHOUT
    eh_without_dir = SAMPLES_DIR / "eval_harness_sample" / "without_harmness_evaluation"
    all_rows.extend(collect_entries(eh_without_dir, "eval_harness_WITHOUT"))

    # Assign row IDs
    for i, row in enumerate(all_rows, 1):
        row["row_id"] = i

    # Write CSV
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Annotation sheet generated: {OUTPUT_CSV}")
    print(f"  Total rows: {len(all_rows)}")

    # Breakdown
    from collections import Counter
    counts = Counter(r["sample_set"] for r in all_rows)
    for k, v in sorted(counts.items()):
        print(f"  {k}: {v} entries")

    print(f"\nAnnotation codes:")
    print(f"  1 = Correct")
    print(f"  2 = Incorrect")
    print(f"  3 = No detail provided")


if __name__ == "__main__":
    main()
