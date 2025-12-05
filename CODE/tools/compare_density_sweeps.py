"""Generate harmonized comparison plots and REV summaries for constant-density sweeps."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple

import math

import matplotlib.pyplot as plt
import pandas as pd

# Relative tolerance for plateau detection once samples are considered converged.
PLATEAU_TOLERANCE = 0.10  # 10 %
# Number of largest-L bins to average for plateau statistics.
PLATEAU_TAIL_BINS = 3

DATA_FILES: Dict[str, Path] = {
    "1e12": Path("CODE/FINAL SET/const_density_1e12/sweep_const_density_1e12_full.csv"),
    "1e13": Path("CODE/FINAL SET/const_density_1e13/sweep_const_density_1e13_full.csv"),
    "1e14": Path("CODE/FINAL SET/const_density_1e14/sweep_const_density_1e14_full.csv"),
}

OUTPUT_DIR = Path("CODE/FINAL SET/plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STATS_COLUMNS = [
    "dir_D_eff",
    "neu_D_eff",
    "tau_dir",
    "tau_neu",
    "porosity",
]


def round_sig(value: float, sig: int = 12) -> float:
    """Round to the requested number of significant digits."""
    if value == 0.0:
        return 0.0
    return round(value, sig - int(math.floor(math.log10(abs(value)))) - 1)


def harmonize_bins(df: pd.DataFrame) -> pd.DataFrame:
    """Add a canonical L bin using significant-digit rounding."""
    canonical = df["L_m"].astype(float).map(lambda v: round_sig(v, 12))
    df = df.copy()
    df["L_bin"] = canonical
    return df


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Compute count, mean, and std for each column grouped by canonical L."""
    grouped = (
        df.groupby("L_bin", sort=True)[STATS_COLUMNS]
        .agg(["count", "mean", "std"])
        .sort_index()
    )
    return grouped


def load_all() -> Dict[str, pd.DataFrame]:
    datasets: Dict[str, pd.DataFrame] = {}
    for label, path in DATA_FILES.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing sweep results for density {label}: {path}")
        df = pd.read_csv(path)
        df = harmonize_bins(df)
        datasets[label] = aggregate(df)
    return datasets


def plot_with_error(ax, x, mean, std, *, label: str, color: str | None = None):
    std = std.fillna(0.0)
    ax.plot(x, mean, marker="o", label=label, color=color)
    ax.fill_between(x, mean - std, mean + std, alpha=0.2, color=color)


def make_plots(stats: Dict[str, pd.DataFrame]) -> None:
    colors = {
        "1e12": "tab:blue",
        "1e13": "tab:orange",
        "1e14": "tab:green",
    }

    # Effective diffusivity (Dirichlet / Neumann)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
    for label, df in stats.items():
        bins = df.index.to_numpy()
        plot_with_error(
            axes[0],
            bins,
            df["dir_D_eff"]["mean"],
            df["dir_D_eff"]["std"],
            label=label,
            color=colors.get(label),
        )
        plot_with_error(
            axes[1],
            bins,
            df["neu_D_eff"]["mean"],
            df["neu_D_eff"]["std"],
            label=label,
            color=colors.get(label),
        )
    axes[0].set_xscale("log")
    axes[1].set_xscale("log")
    axes[0].set_ylabel("Dirichlet $D_{eff}$")
    axes[0].set_xlabel("L [m]")
    axes[1].set_ylabel("Neumann $D_{eff}$")
    axes[1].set_xlabel("L [m]")
    axes[0].legend(title="Density")
    axes[0].grid(True, which="both", linestyle=":", linewidth=0.5)
    axes[1].grid(True, which="both", linestyle=":", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "diffusivity_vs_L.png", dpi=300)

    # Tortuosity plots (Dirichlet / Neumann)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
    for label, df in stats.items():
        bins = df.index.to_numpy()
        plot_with_error(
            axes[0],
            bins,
            df["tau_dir"]["mean"],
            df["tau_dir"]["std"],
            label=label,
            color=colors.get(label),
        )
        plot_with_error(
            axes[1],
            bins,
            df["tau_neu"]["mean"],
            df["tau_neu"]["std"],
            label=label,
            color=colors.get(label),
        )
    axes[0].set_xscale("log")
    axes[1].set_xscale("log")
    axes[0].set_ylabel("Dirichlet $\\tau$")
    axes[0].set_xlabel("L [m]")
    axes[1].set_ylabel("Neumann $\\tau$")
    axes[1].set_xlabel("L [m]")
    axes[0].legend(title="Density")
    axes[0].grid(True, which="both", linestyle=":", linewidth=0.5)
    axes[1].grid(True, which="both", linestyle=":", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "tortuosity_vs_L.png", dpi=300)

    # Porosity plot
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, df in stats.items():
        bins = df.index.to_numpy()
        plot_with_error(
            ax,
            bins,
            df["porosity"]["mean"],
            df["porosity"]["std"],
            label=label,
            color=colors.get(label),
        )
    ax.set_xscale("log")
    ax.set_ylabel("Porosity")
    ax.set_xlabel("L [m]")
    ax.legend(title="Density")
    ax.grid(True, which="both", linestyle=":", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "porosity_vs_L.png", dpi=300)

    plt.close("all")


def _plateau_index(series: pd.Series, final_mean: float) -> int | None:
    """Return the index of the first bin within tolerance of the final mean and remaining there."""
    values = series.to_numpy()
    for idx in range(len(values)):
        diff = abs(values[idx] - final_mean) / max(final_mean, 1e-12)
        if diff <= PLATEAU_TOLERANCE:
            # Ensure all subsequent bins remain within tolerance.
            remaining = values[idx:]
            if all(abs(v - final_mean) / max(final_mean, 1e-12) <= PLATEAU_TOLERANCE for v in remaining):
                return idx
    return None


def compute_plateau_stats(stats: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, Tuple[float, float, float]]]:
    """Extract REV threshold and plateau mean/std for each density."""
    plateau_summary: Dict[str, Dict[str, Tuple[float, float, float]]] = {}
    for label, df in stats.items():
        result: Dict[str, Tuple[float, float, float]] = {}
        bins = df.index.to_numpy()
        tail_slice = slice(-PLATEAU_TAIL_BINS, None)
        tail_bins = bins[tail_slice]

        def tail_stats(column: str) -> Tuple[float, float]:
            tail_mean = df[column]["mean"].iloc[tail_slice].mean()
            tail_std = df[column]["mean"].iloc[tail_slice].std(ddof=0)
            return tail_mean, tail_std

        dir_mean = df["dir_D_eff"]["mean"]
        dir_tail_mean, dir_tail_std = tail_stats("dir_D_eff")
        tolerable_idx = _plateau_index(dir_mean, dir_tail_mean)
        L_plateau = float("nan") if tolerable_idx is None else bins[tolerable_idx]

        result["dir_D_eff"] = (L_plateau, dir_tail_mean, dir_tail_std)

        tau_mean = df["tau_dir"]["mean"]
        tau_tail_mean, tau_tail_std = tail_stats("tau_dir")
        tau_idx = _plateau_index(tau_mean, tau_tail_mean)
        tau_plateau = float("nan") if tau_idx is None else bins[tau_idx]
        result["tau_dir"] = (tau_plateau, tau_tail_mean, tau_tail_std)

        por_mean = df["porosity"]["mean"]
        por_tail_mean, por_tail_std = tail_stats("porosity")
        por_idx = _plateau_index(por_mean, por_tail_mean)
        por_plateau = float("nan") if por_idx is None else bins[por_idx]
        result["porosity"] = (por_plateau, por_tail_mean, por_tail_std)

        plateau_summary[label] = result
    return plateau_summary


def format_plateau_summary(summary: Dict[str, Dict[str, Tuple[float, float, float]]]) -> str:
    lines = ["Density,Metric,REV_L_plateau,Mean,Std"]
    for density, metrics in summary.items():
        for metric, (plateau_L, mean_val, std_val) in metrics.items():
            lines.append(
                f"{density},{metric},{plateau_L:.6g},{mean_val:.6g},{std_val:.6g}"
            )
    return "\n".join(lines)


def main() -> None:
    stats = load_all()
    make_plots(stats)
    summary = compute_plateau_stats(stats)
    csv_output = OUTPUT_DIR / "rev_plateau_summary.csv"
    csv_output.write_text(format_plateau_summary(summary))
    print("Generated plots:")
    for name in ("diffusivity_vs_L.png", "tortuosity_vs_L.png", "porosity_vs_L.png"):
        print(f" - {OUTPUT_DIR / name}")
    print("Plateau summary: ", csv_output)


if __name__ == "__main__":
    main()
