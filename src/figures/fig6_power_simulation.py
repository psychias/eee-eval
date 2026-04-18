"""
fig6_power_simulation.py — two-panel power figure for harness_differs.

Left panel:  power as a function of k (number of sources) per benchmark,
             with R² ± 0.15 uncertainty bands (shaded); 80% target dashed.
Right panel: minimum sources needed for 80% power per benchmark,
             with error bars spanning the R² ± 0.15 range.

Falls back to global power_simulation.csv if per-benchmark data unavailable.

Output: figures/fig6_power_simulation.pdf + submission/fig6_power_simulation.png
"""
from __future__ import annotations
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

OUT_DIR = _ROOT / "figures"
OUT_DIR.mkdir(exist_ok=True)
SUB_DIR = _ROOT / "submission"
SUB_DIR.mkdir(exist_ok=True)

# Colorblind-friendly palette (Wong 2011) + distinct markers per benchmark
BENCH_COLORS = {
    "GSM8K":      "#0072B2",   # blue
    "HumanEval":  "#E69F00",   # orange
    "MMLU":       "#009E73",   # green
    "HellaSwag":  "#D55E00",   # vermillion
    "MATH":       "#CC79A7",   # pink
    "ARC-Challenge": "#F0E442", # yellow
    "TruthfulQA": "#56B4E9",   # sky blue
}
BENCH_MARKERS = {
    "GSM8K":      "o",
    "HumanEval":  "s",
    "MMLU":       "D",
    "HellaSwag":  "^",
    "MATH":       "v",
    "ARC-Challenge": "P",
    "TruthfulQA": "X",
}


def main():
    bench_csv = _ROOT / "analysis_output" / "power_simulation_per_bench.csv"
    global_csv = _ROOT / "analysis_output" / "power_simulation.csv"

    has_bench = bench_csv.exists()
    if has_bench:
        bdf = pd.read_csv(bench_csv)
        benchmarks = sorted(bdf["benchmark"].unique().tolist())
    else:
        print("No per-benchmark data; using global power curve only.",
              file=sys.stderr)

    if not has_bench and not global_csv.exists():
        print("Run power_simulation.py first.", file=sys.stderr)
        sys.exit(1)

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 5),
                                             gridspec_kw={"width_ratios": [2, 1]})

    # ── LEFT PANEL: power vs k per benchmark ──
    if has_bench:
        for bench in benchmarks:
            sub = bdf[bdf["benchmark"] == bench]
            # Main power curve (central estimate, not shifted)
            main = sub[sub["r2_shift"] == "lo"].copy()  # both lo/hi have same mean_power
            main = main.sort_values("k")
            k_vals = main["k"].values
            power_vals = main["mean_power"].values

            # R²±0.15 bands
            lo_rows = sub[sub["r2_shift"] == "lo"].sort_values("k")
            hi_rows = sub[sub["r2_shift"] == "hi"].sort_values("k")
            band_lo = lo_rows["power_shifted"].values
            band_hi = hi_rows["power_shifted"].values

            color = BENCH_COLORS.get(bench, "#999999")
            marker = BENCH_MARKERS.get(bench, "o")
            ax_left.plot(k_vals, power_vals, color=color, linewidth=2,
                         marker=marker, markersize=6, markeredgecolor="white",
                         markeredgewidth=0.5, label=bench, zorder=3)
            # More prominent uncertainty bands: translucent fill with visible boundary
            band_min = np.minimum(band_lo, band_hi)
            band_max = np.maximum(band_lo, band_hi)
            ax_left.fill_between(k_vals, band_min, band_max,
                                 alpha=0.20, color=color, zorder=1)
            ax_left.plot(k_vals, band_min, color=color, linewidth=0.5,
                         alpha=0.4, linestyle=":", zorder=2)
            ax_left.plot(k_vals, band_max, color=color, linewidth=0.5,
                         alpha=0.4, linestyle=":", zorder=2)
    else:
        gdf = pd.read_csv(global_csv)
        k_vals = gdf["k"].values
        power_vals = gdf["mean_power"].values
        ci_lo = gdf["ci_lo_95"].values
        ci_hi = gdf["ci_hi_95"].values
        ax_left.plot(k_vals, power_vals, "o-", color="#4C72B0", linewidth=2,
                     markersize=5, label="All benchmarks")
        ax_left.fill_between(k_vals, ci_lo, ci_hi, alpha=0.2, color="#4C72B0")

    ax_left.axhline(0.80, color="#C44E52", linestyle="--", linewidth=1.5,
                    label="80% power target", zorder=2)
    ax_left.set_xlabel("Number of independent evaluation sources (k)", fontsize=10)
    ax_left.set_ylabel("Statistical power", fontsize=10)
    ax_left.set_title("Power vs. Number of Sources", fontsize=11, fontweight="bold")
    ax_left.set_ylim(-0.05, 1.05)
    ax_left.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax_left.xaxis.set_major_locator(mticker.MultipleLocator(2))
    ax_left.legend(fontsize=8, loc="upper left", framealpha=0.9)
    ax_left.grid(True, alpha=0.3, linestyle="--")

    # ── RIGHT PANEL: minimum k for 80% power per benchmark ──
    if has_bench:
        k80_data = {}
        k80_lo = {}
        k80_hi = {}
        for bench in benchmarks:
            sub = bdf[bdf["benchmark"] == bench]
            main = sub[sub["r2_shift"] == "lo"].sort_values("k")
            lo_rows = sub[sub["r2_shift"] == "lo"].sort_values("k")
            hi_rows = sub[sub["r2_shift"] == "hi"].sort_values("k")

            # Find k where power ≥ 0.80
            def find_k80(k_arr, pwr_arr):
                for k, p in zip(k_arr, pwr_arr):
                    if p >= 0.80:
                        return k
                return k_arr[-1] + 2  # beyond range

            k80_data[bench] = find_k80(main["k"].values, main["mean_power"].values)
            k80_lo[bench]   = find_k80(lo_rows["k"].values, lo_rows["power_shifted"].values)
            k80_hi[bench]   = find_k80(hi_rows["k"].values, hi_rows["power_shifted"].values)

        bench_names = list(k80_data.keys())
        k80_vals = [k80_data[b] for b in bench_names]
        err_lo = [max(0.0, k80_data[b] - min(k80_lo[b], k80_hi[b])) for b in bench_names]
        err_hi = [max(0.0, max(k80_lo[b], k80_hi[b]) - k80_data[b]) for b in bench_names]
        # Color-code by reachability: green if ≤20, amber if 20-30, red if >30
        colors = []
        for b in bench_names:
            k = k80_data[b]
            if k <= 20:
                colors.append("#27AE60")  # reachable (green)
            elif k <= 30:
                colors.append("#F5B041")  # marginal (amber)
            else:
                colors.append("#E74C3C")  # out of range (red)

        y_pos = range(len(bench_names))
        ax_right.barh(y_pos, k80_vals, color=colors, alpha=0.85, height=0.6,
                      edgecolor="white")
        ax_right.errorbar(k80_vals, y_pos,
                          xerr=[err_lo, err_hi],
                          fmt="none", ecolor="black", capsize=4, linewidth=1.2)
        ax_right.set_yticks(y_pos)
        ax_right.set_yticklabels(bench_names, fontsize=9)
        ax_right.set_xlabel("Min. sources for 80% power", fontsize=10)
        ax_right.set_title("Sources Needed", fontsize=11, fontweight="bold")
        ax_right.axvline(20, color="#AAAAAA", linestyle=":", linewidth=1,
                         label="k = 20 (max simulated)")
        ax_right.grid(True, axis="x", alpha=0.3, linestyle="--")
        ax_right.invert_yaxis()
    else:
        gdf = pd.read_csv(global_csv)
        k80 = gdf.loc[gdf["mean_power"] >= 0.80, "k"]
        k80_val = k80.min() if len(k80) > 0 else 20
        ax_right.barh(["All"], [k80_val], color="#4C72B0", alpha=0.8, height=0.4)
        ax_right.set_xlabel("Min. sources for 80% power", fontsize=10)
        ax_right.set_title("Sources Needed", fontsize=11, fontweight="bold")
        ax_right.grid(True, axis="x", alpha=0.3, linestyle="--")

    fig.suptitle("Power Simulation for the Harness-Differs Predictor",
                 fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()

    out_pdf = OUT_DIR / "fig6_power_simulation.pdf"
    out_png = SUB_DIR / "fig6_power_simulation.png"
    fig.savefig(out_pdf, bbox_inches="tight", dpi=150)
    fig.savefig(out_png, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved → {out_pdf}")
    print(f"Saved → {out_png}")


if __name__ == "__main__":
    main()
