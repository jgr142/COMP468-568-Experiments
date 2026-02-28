#!/usr/bin/env python3
import argparse
import pandas as pd
import matplotlib.pyplot as plt


EXPECTED_8COLS = ["impl", "d1", "d2", "d3", "batch", "activation", "time_ms", "gflops"]


def read_benchmark_csv(path: str) -> pd.DataFrame:
    """
    Handles two cases:
      A) Proper CSV with correct headers (unknown names)
      B) Your pasted case: header has 6 cols but rows have 8 values:
         baseline,512,512,512,64,relu,17.86,3.76
    In case (B), we re-read with EXPECTED_8COLS.
    """
    df = pd.read_csv(path)

    # If pandas read it, but the data clearly has 8 columns worth of values,
    # you might see weird column names and/or misalignment.
    # We'll detect by checking number of columns and whether required fields exist.
    required = {"impl", "batch"}
    has_impl_batch = required.issubset(set(df.columns))

    if len(df.columns) != 8 or not has_impl_batch or ("time_ms" not in df.columns) or ("gflops" not in df.columns):
        # Re-read as headerless and assign the correct 8 columns.
        df = pd.read_csv(path, header=None, names=EXPECTED_8COLS)

        # If the original file had a header line, it will now appear as a data row;
        # drop it if detected.
        if df.loc[0, "impl"] == "impl":
            df = df.iloc[1:].reset_index(drop=True)

    # Coerce types
    df["batch"] = pd.to_numeric(df["batch"], errors="raise")
    df["time_ms"] = pd.to_numeric(df["time_ms"], errors="raise")
    df["gflops"] = pd.to_numeric(df["gflops"], errors="raise")
    # d1/d2/d3 are dims; keep numeric if present
    for c in ["d1", "d2", "d3"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def plot_one_group(group: pd.DataFrame, title_suffix: str, out_prefix: str | None):
    # Ensure both impls are there
    impls = set(group["impl"].unique())
    if not {"baseline", "activation_fused"}.issubset(impls):
        print(f"Warning: missing baseline/fused in group {title_suffix}: found {sorted(impls)}")

    # Sort for plotting
    group = group.sort_values(["impl", "batch"])

    # ---- time plot ----
    plt.figure()
    for impl, g in group.groupby("impl"):
        g = g.sort_values("batch")
        plt.plot(g["batch"], g["time_ms"], marker="o", label=impl)
    plt.title(f"Time vs Batch {title_suffix}")
    plt.xlabel("batch")
    plt.ylabel("time_ms")
    plt.grid(True)
    plt.legend()

    if out_prefix:
        plt.savefig(f"{out_prefix}{title_suffix}_time.png".replace(" ", "_"), dpi=200, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    # ---- gflops plot ----
    plt.figure()
    for impl, g in group.groupby("impl"):
        g = g.sort_values("batch")
        plt.plot(g["batch"], g["gflops"], marker="o", label=impl)
    plt.title(f"GFLOPS vs Batch {title_suffix}")
    plt.xlabel("batch")
    plt.ylabel("gflops")
    plt.grid(True)
    plt.legend()

    if out_prefix:
        plt.savefig(f"{out_prefix}{title_suffix}_gflops.png".replace(" ", "_"), dpi=200, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    # ---- speedup plot (baseline_time / fused_time) ----
    b = group[group["impl"] == "baseline"][["batch", "time_ms"]].set_index("batch")
    f = group[group["impl"] == "activation_fused"][["batch", "time_ms"]].set_index("batch")
    joined = b.join(f, lsuffix="_baseline", rsuffix="_fused", how="inner")
    if not joined.empty:
        speedup = joined["time_ms_baseline"] / joined["time_ms_fused"]

        plt.figure()
        plt.plot(speedup.index, speedup.values, marker="o")
        plt.title(f"Speedup (baseline/fused) {title_suffix}")
        plt.xlabel("batch")
        plt.ylabel("speedup")
        plt.grid(True)

        if out_prefix:
            plt.savefig(f"{out_prefix}{title_suffix}_speedup.png".replace(" ", "_"), dpi=200, bbox_inches="tight")
            plt.close()
        else:
            plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="Path to CSV (your benchmark output).")
    ap.add_argument("--out-prefix", default=None,
                    help="If set, saves PNGs with this prefix instead of showing plots.")
    args = ap.parse_args()

    df = read_benchmark_csv(args.csv)

    # If we have d1/d2/d3, group by those dims (that matches your rows: 512,512,512 etc.)
    dim_cols = [c for c in ["d1", "d2", "d3"] if c in df.columns and df[c].notna().any()]

    if dim_cols:
        for dims, g in df.groupby(dim_cols):
            if not isinstance(dims, tuple):
                dims = (dims,)
            suffix = " dims=" + "x".join(str(int(x)) for x in dims)
            plot_one_group(g, suffix, args.out_prefix)
    else:
        # Fallback: group by whatever "layers" exists, else plot all together
        if "layers" in df.columns:
            for lv, g in df.groupby("layers"):
                suffix = f" layers={lv}"
                plot_one_group(g, suffix, args.out_prefix)
        else:
            plot_one_group(df, "", args.out_prefix)


if __name__ == "__main__":
    main()
