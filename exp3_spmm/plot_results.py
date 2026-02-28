#!/usr/bin/env python3
import os
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "spmm_results.csv"
OUT_DIR = "plots_spmm"

def ensure_out_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def save_fig(path: str) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

def main():
    df = pd.read_csv(CSV_PATH)

    # Basic sanity / types
    for c in ["M", "K", "N", "density", "nnz", "time_ms", "gflops"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["impl"] = df["impl"].astype(str)

    ensure_out_dir(OUT_DIR)

    # ---- 1) Per-(M,K) curves: GFLOP/s vs N and Time vs N ----
    groups = df.groupby(["M", "K", "density"], dropna=False)

    for (M, K, density), g in groups:
        # pivot to align baseline/optimized by N
        g2 = g.pivot_table(index="N", columns="impl", values=["gflops", "time_ms"], aggfunc="mean")
        g2 = g2.sort_index()

        # GFLOP/s vs N
        plt.figure()
        for impl in ["baseline", "optimized"]:
            if ("gflops", impl) in g2.columns:
                plt.plot(g2.index, g2[("gflops", impl)], marker="o", label=impl)
        plt.xlabel("N")
        plt.ylabel("GFLOP/s")
        plt.title(f"SpMM GFLOP/s vs N (M={M}, K={K}, density={density})")
        plt.legend()
        save_fig(os.path.join(OUT_DIR, f"gflops_vs_N_M{M}_K{K}_d{density}.png"))

        # Time vs N
        plt.figure()
        for impl in ["baseline", "optimized"]:
            if ("time_ms", impl) in g2.columns:
                plt.plot(g2.index, g2[("time_ms", impl)], marker="o", label=impl)
        plt.xlabel("N")
        plt.ylabel("Time (ms)")
        plt.title(f"SpMM Time vs N (M={M}, K={K}, density={density})")
        plt.legend()
        save_fig(os.path.join(OUT_DIR, f"time_vs_N_M{M}_K{K}_d{density}.png"))

        # Speedup vs N (baseline / optimized)
        if ("time_ms", "baseline") in g2.columns and ("time_ms", "optimized") in g2.columns:
            speedup = g2[("time_ms", "baseline")] / g2[("time_ms", "optimized")]
            plt.figure()
            plt.plot(g2.index, speedup, marker="o")
            plt.axhline(1.0, linestyle="--")
            plt.xlabel("N")
            plt.ylabel("Speedup (baseline / optimized)")
            plt.title(f"SpMM Speedup vs N (M={M}, K={K}, density={density})")
            save_fig(os.path.join(OUT_DIR, f"speedup_vs_N_M{M}_K{K}_d{density}.png"))

    # ---- 2) Overall scatter: baseline vs optimized GFLOP/s ----
    # Join rows by (M,K,N,density,nnz) assuming both impls exist for each config
    key_cols = ["M", "K", "N", "density", "nnz"]
    base = df[df["impl"] == "baseline"].set_index(key_cols)
    opt = df[df["impl"] == "optimized"].set_index(key_cols)

    joined = base[["gflops", "time_ms"]].join(
        opt[["gflops", "time_ms"]],
        how="inner",
        lsuffix="_baseline",
        rsuffix="_optimized",
    ).reset_index()

    if not joined.empty:
        plt.figure()
        plt.scatter(joined["gflops_baseline"], joined["gflops_optimized"])
        mn = min(joined["gflops_baseline"].min(), joined["gflops_optimized"].min())
        mx = max(joined["gflops_baseline"].max(), joined["gflops_optimized"].max())
        plt.plot([mn, mx], [mn, mx], linestyle="--")  # y=x
        plt.xlabel("Baseline GFLOP/s")
        plt.ylabel("Optimized GFLOP/s")
        plt.title("Baseline vs Optimized GFLOP/s (all configs)")
        save_fig(os.path.join(OUT_DIR, "scatter_gflops_baseline_vs_optimized.png"))

        # Optional: histogram of speedups
        speedup = joined["time_ms_baseline"] / joined["time_ms_optimized"]
        plt.figure()
        plt.hist(speedup, bins=30)
        plt.xlabel("Speedup (baseline / optimized)")
        plt.ylabel("Count")
        plt.title("Distribution of Speedups (all configs)")
        save_fig(os.path.join(OUT_DIR, "hist_speedup.png"))

    print(f"Saved plots to: {OUT_DIR}/")

if __name__ == "__main__":
    main()
