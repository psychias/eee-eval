"""Write all cross-resource collision pairs to collision.txt

Fast version: loads leaderboard data from aggregated CSV and only walks
arxiv_extraction_general/llm/ for paper data (avoids full data/ tree walk).
"""
import json, os, re, sys
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def normalize_model_id(raw):
    s = str(raw).strip()
    # Strip HF org prefix: 'meta-llama/Llama-3.1-8B' -> 'Llama-3.1-8B'
    if "/" in s:
        s = s.rsplit("/", 1)[-1]
    s = re.sub(r"\s*\((?:Prompt|FC|Chat)\)\s*$", "", s, flags=re.IGNORECASE)
    s = s.lower()
    # Normalize underscores to hyphens (papers use Gemma_2_9B, HF uses Gemma-2-9B)
    s = s.replace("_", "-")
    s = re.sub(r"\s+", "-", s)
    return s


def normalize_benchmark(raw):
    s = str(raw).strip()
    # Fix stale names from pre-audit JSON files
    _fixup = {
        "ARC-C": "ARC-Challenge", "ARC-E": "ARC-Easy",
        "ARC-c": "ARC-Challenge", "ARC-e": "ARC-Easy",
        # BigBench-Hard variants → BBH
        "BigBench-Hard": "BBH", "bigbench-hard": "BBH",
        "Big-Bench Hard": "BBH", "big-bench hard": "BBH",
        "Big-Bench-Hard": "BBH", "big-bench-hard": "BBH",
        "Bigbench Hard": "BBH", "bigbench hard": "BBH",
        # GPQA variants → canonical
        "GPQA Diamond": "GPQA-Diamond", "GPQA-D": "GPQA-Diamond",
        "gpqa diamond": "GPQA-Diamond", "gpqa-d": "GPQA-Diamond",
    }
    s = _fixup.get(s, s)
    s = s.lower()
    s = re.sub(r"\s*\(\d+-shot\)\s*$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+\d+-shot\s*$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s).strip()
    # Alias lookup (after lowercasing)
    s_lookup = s.replace("-", " ").replace("_", " ").strip()
    _aliases = {
        "bigbench hard": "bbh",
        "big bench hard": "bbh",
        "gpqa diamond": "gpqa-diamond",
        "gpqa d": "gpqa-diamond",
    }
    s = _aliases.get(s_lookup, s)
    return s

SOURCE_LABELS = {
    "open_llm_leaderboard_v2": "OLv2",
    "alpacaeval2": "AlpacaEval2",
    "chatbot_arena": "Arena",
    "bigcodebench": "BigCodeBench",
    "evalplus": "EvalPlus",
    "bfcl": "BFCL",
    "wildbench": "WildBench",
    "swe_bench": "SWE-bench",
    "mt_bench": "MT-Bench",
    "hf_model_card": "HF-Card",
    "arxiv_html_llm": "ArXiv-LLM",
}

def resource_label(r):
    if r in SOURCE_LABELS:
        return SOURCE_LABELS[r]
    return r.replace("_", " ")[:50]



def main():
    # ── Load leaderboard data from aggregated CSV (fast) ──
    print("Loading leaderboard data from CSV...")
    csv_path = os.path.join(ROOT, "data", "aggregated", "all_results.csv")
    lb = pd.read_csv(csv_path, low_memory=False)
    lb = lb[lb["source"] != "papers_with_code"].reset_index(drop=True)
    lb["resource"] = lb["source"]  # each leaderboard = one resource
    print(f"  Leaderboard records: {len(lb):,}")

    # ── Load arxiv paper data (walk only arxiv dir) ──
    print("Loading arxiv extraction data...")
    arxiv_dir = os.path.join(ROOT, "data", "arxiv_extraction_general", "llm")
    rows = []
    for dirpath, dirnames, filenames in os.walk(arxiv_dir):
        for fn in filenames:
            if not fn.endswith(".json"):
                continue
            fp = os.path.join(dirpath, fn)
            try:
                with open(fp, "r", encoding="utf-8") as fh:
                    rec = json.load(fh)
                mi = rec.get("model_info", {})
                src_meta = rec.get("source_metadata", {})
                ad = src_meta.get("additional_details", {})
                paper_title = ad.get("paper_title", "")
                # resource = paper directory name
                rel = os.path.relpath(dirpath, arxiv_dir)
                paper_dir = rel.split(os.sep)[0] if os.sep in rel else rel
                for er in rec.get("evaluation_results", []):
                    rows.append({
                        "model_id": mi.get("name", ""),
                        "benchmark": er.get("evaluation_name", ""),
                        "score": er.get("score_details", {}).get("score", ""),
                        "source": src_meta.get("source_name", "arxiv_html_llm"),
                        "resource": paper_dir,
                    })
            except Exception:
                continue

    arxiv_df = pd.DataFrame(rows)
    print(f"  Arxiv records: {len(arxiv_df):,}")

    # ── Combine ──
    lb_sub = pd.DataFrame({
        "model_id": lb["model"].values,
        "benchmark": lb["benchmark"].values,
        "score": lb["score"].values,
        "source": lb["source"].values,
        "resource": lb["source"].values,
    })
    df = pd.concat([lb_sub, arxiv_df], ignore_index=True)
    print(f"  Total records: {len(df):,}")

    # ── Fix OLv2 GPQA-Diamond metric mismatch ─────────────────────
    # OLv2 uses acc_norm (max ~29) for GPQA-Diamond but papers use raw accuracy
    # (up to 92). Relabel so they don't create spurious cross-source collisions.
    _gpqa_olv2 = (
        (df["benchmark"] == "GPQA-Diamond") &
        (df["source"] == "open_llm_leaderboard_v2")
    )
    n_gpqa = _gpqa_olv2.sum()
    df.loc[_gpqa_olv2, "benchmark"] = "GPQA-Diamond (acc_norm)"
    print(f"  Relabeled {n_gpqa:,} OLv2 GPQA-Diamond → GPQA-Diamond (acc_norm)")

    # ── Score normalization (0-1 -> 0-100) ──
    print("Normalizing scores...")
    df["score_num"] = pd.to_numeric(df["score"], errors="coerce")
    df["benchmark_norm"] = df["benchmark"].apply(normalize_benchmark)

    n_rescaled = 0
    for bench in df["benchmark_norm"].unique():
        mask = df["benchmark_norm"] == bench
        bench_scores = df.loc[mask, "score_num"].dropna()
        if len(bench_scores) < 2:
            continue
        frac_above = (bench_scores.abs() > 1.0).sum() / len(bench_scores)
        n_in_01 = ((bench_scores >= 0) & (bench_scores <= 1.0)).sum()
        if frac_above >= 0.6 and n_in_01 > 0:
            rescale_mask = mask & (df["score_num"] >= 0) & (df["score_num"] <= 1.0)
            n_rescaled += rescale_mask.sum()
            df.loc[rescale_mask, "score_num"] = df.loc[rescale_mask, "score_num"] * 100

    print(f"  Rescaled {n_rescaled} scores")

    # ── Normalize model names ──
    df["model_norm"] = df["model_id"].apply(normalize_model_id)

    leaderboard_resources = set(
        df[~df["source"].str.contains("arxiv", case=False, na=False)]["resource"].unique()
    )

    # ── Aggregate per resource ──
    print("Finding collisions...")
    per_resource = (
        df.dropna(subset=["score_num"])
        .groupby(["model_norm", "benchmark_norm", "resource"])
        .agg(score_median=("score_num", "median"), n_records=("score_num", "size"), source=("source", "first"))
        .reset_index()
    )

    pair_counts = per_resource.groupby(["model_norm", "benchmark_norm"])["resource"].nunique()
    multi = pair_counts[pair_counts >= 2].index

    collision_rows = []
    for (mn, bn) in multi:
        grp = per_resource[(per_resource["model_norm"] == mn) & (per_resource["benchmark_norm"] == bn)].sort_values("score_median")
        scores = grp["score_median"].values
        score_range = scores.max() - scores.min()
        if score_range <= 0.01:
            continue
        details = []
        for _, row in grp.iterrows():
            details.append({
                "resource": row["resource"], "source": row["source"],
                "score": row["score_median"], "n_records": int(row["n_records"]),
            })
        collision_rows.append({
            "model": mn, "benchmark": bn, "n_resources": len(grp),
            "score_range": score_range, "score_min": scores.min(), "score_max": scores.max(),
            "details": details,
        })

    collision_rows.sort(key=lambda x: -x["score_range"])
    print(f"  Found {len(collision_rows)} collision pairs")

    # ── Write collision.txt ──
    out_path = os.path.join(ROOT, "analysis_output", "collision.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("=" * 100 + "\n")
        f.write("  ALL CROSS-RESOURCE COLLISION PAIRS\n")
        f.write(f"  Total: {len(collision_rows)}\n")
        f.write(f"  (Each arxiv paper and each leaderboard treated as a separate resource)\n")
        f.write(f"  Scores normalized: {n_rescaled} values rescaled from 0-1 to 0-100\n")
        f.write("=" * 100 + "\n\n")

        for idx, c in enumerate(collision_rows, 1):
            f.write(f"--- Collision #{idx} ---\n")
            f.write(f"  Model:      {c['model']}\n")
            f.write(f"  Benchmark:  {c['benchmark']}\n")
            f.write(f"  Resources:  {c['n_resources']}\n")
            f.write(f"  Score range: {c['score_range']:.2f}  (min={c['score_min']:.2f}, max={c['score_max']:.2f})\n")
            f.write(f"  Per-resource scores:\n")
            for d in sorted(c["details"], key=lambda x: x["score"]):
                lbl = resource_label(d["resource"])
                tag = "leaderboard" if d["resource"] in leaderboard_resources else "arxiv paper"
                f.write(f"    {d['score']:>8.2f}  [{tag:<14s}]  {lbl}  (n={d['n_records']})\n")
            f.write("\n")

    print(f"\nDone! Written to {out_path}")




if __name__ == "__main__":
    main()
