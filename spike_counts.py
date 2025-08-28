"""This creates 3 graphs for spike counts

  1) Macro A1-D6 (subsites collapsed): absolute + per-device relative
  2) Per-letter subsites: absolute + per-device relative
  3) All subsites clustered: absolute + per-device relative

To reuse my code, I have installed the following depenencies:
Pandas, Numpy, Matplotlib, Scipy 

Author: Tracy Huang
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
from scipy.cluster.hierarchy import linkage, leaves_list

#SETTINGS
CSV_PATH = Path("/Users/tracy/Downloads/Bai Lab Downloads/spike_counts.csv")
BIN_SEC  = 180
OUTDIR   = Path("heatmap_outputs"); OUTDIR.mkdir(exist_ok=True)

# Color and Contrast
CMAP = "turbo"                 # high-contrast: 'turbo'|'viridis'|'magma'|'inferno'
USE_PERCENTILE_CLIP = True     # clip color scale to percentiles
PCTL_MIN, PCTL_MAX   = 5, 99.5 # adjust if needed
USE_LOG_SCALE        = False   # True helps when a few channels dominate
LOG_EPS              = 1.0     # added before log

# Layout and labels
THIN_X_LABELS_STEP   = 0       # 0/1 show all and 2 show every other
WIDTH_SCALE          = 0.8     # increase if needed to make columns wider



#Data Scaling Functions
def load_and_clean(csv_path: Path) -> pd.DataFrame:
    """This cleans the data and makes sure the columns 
    for the times are in numeric integer types"""

    df = pd.read_csv(csv_path)
    for col in ["Interval Start (S)", "Interval End (S)"]:
        if col not in df.columns:
            raise ValueError(f'Missing column: {col}')
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[df["Interval Start (S)"].notna() & df["Interval End (S)"].notna()].copy()
    df["Interval Start (S)"] = df["Interval Start (S)"].astype(int)
    df["Interval End (S)"]   = df["Interval End (S)"].astype(int)
    return df


def detect_device_columns(df: pd.DataFrame) -> list[str]:
    """Filters out the non electrodes columns, like investigator, 
    interval start and ends, and unnamed variables"""

    non = {"Investigator","KELLY","Interval Start (S)","Interval End (S)","Unnamed: 4"}
    return [c for c in df.columns if c not in non and not str(c).startswith("Unnamed")]

def coerce_numeric(df: pd.DataFrame, cols: list[str]) -> None:
    if cols:
        df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")

def sum_numeric(frame: pd.DataFrame, cols: list[str]) -> pd.Series:
    if not cols:
        return pd.Series(np.zeros(len(frame)), index=frame.index)
    return frame[cols].apply(pd.to_numeric, errors="coerce").sum(axis=1).fillna(0.0)

def bin_and_sum(df: pd.DataFrame, device_cols: list[str], bin_sec: int) -> pd.DataFrame:
    tmp = df.copy()
    tmp["time_bin"] = (tmp["Interval Start (S)"] // bin_sec) * bin_sec
    grouped = tmp.groupby("time_bin", sort=True)[device_cols].sum().fillna(0.0)
    return grouped

def make_y_labels(time_bins: pd.Index, bin_sec: int) -> list[str]:
    return [f"{t//60}–{(t+bin_sec)//60} min" for t in time_bins]

def per_device_minmax(mat: np.ndarray) -> np.ndarray:
    """Scale each column to [0,1] so within-electrode changes pop over time."""
    m = np.asarray(mat, dtype=float)
    col_min = np.nanmin(m, axis=0, keepdims=True)
    col_max = np.nanmax(m, axis=0, keepdims=True)
    rng = np.where((col_max - col_min) == 0, 1.0, (col_max - col_min))
    return (m - col_min) / rng


#Color Scaling Functions
def build_norm(mat: np.ndarray):
    m = np.asarray(mat, dtype=float)
    if USE_LOG_SCALE:
        m = m + LOG_EPS
        vmin = np.percentile(m, PCTL_MIN) if USE_PERCENTILE_CLIP else np.nanmin(m)
        vmax = np.percentile(m, PCTL_MAX) if USE_PERCENTILE_CLIP else np.nanmax(m)
        vmin = max(vmin, LOG_EPS)
        if not np.isfinite(vmax) or vmax <= vmin: vmax = vmin + 1e-12
        return LogNorm(vmin=vmin, vmax=vmax), "Total spikes (log)"
    else:
        vmin = np.percentile(m, PCTL_MIN) if USE_PERCENTILE_CLIP else np.nanmin(m)
        vmax = np.percentile(m, PCTL_MAX) if USE_PERCENTILE_CLIP else np.nanmax(m)
        if not np.isfinite(vmin): vmin = 0.0
        if not np.isfinite(vmax) or vmax <= vmin: vmax = vmin + 1e-12
        return Normalize(vmin=vmin, vmax=vmax), "Total spikes (per 3 min)"


#Plotting functions
def thin_labels(labels: list[str], step: int) -> list[str]:
    if step and step > 1:
        return [lab if i % step == 0 else "" for i, lab in enumerate(labels)]
    return labels

def heatmap_cellwise(
    mat: np.ndarray,
    x_labels: list[str],
    y_labels: list[str],
    title: str,
    outpath: Path,
    cmap_name: str = CMAP,
    norm=None,
    cbar_label: str = "",
    width_scale: float = WIDTH_SCALE,
):
    mat = np.asarray(mat, dtype=float)
    n_rows, n_cols = mat.shape

    fig_w = max(18, min(40, n_cols * width_scale))   # wider columns
    fig_h = max(7,  min(26, n_rows * 1.0))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    if norm is None:
        norm, cbar_label = build_norm(mat)

    data = mat + (LOG_EPS if isinstance(norm, LogNorm) else 0.0)
    im = ax.imshow(data, aspect='auto', interpolation='nearest',
                   cmap=plt.get_cmap(cmap_name), norm=norm)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)

    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(thin_labels(x_labels, THIN_X_LABELS_STEP), rotation=90, fontsize=7)
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(y_labels)

    # subtle grid
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which='minor', color='w', linewidth=0.35)
    ax.tick_params(which='minor', length=0)

    ax.set_xlabel("Devices")
    ax.set_ylabel("Time (minutes)")
    ax.set_title(title)
    ax.invert_yaxis() 
    plt.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    print(f"[saved] {outpath}")
    plt.show()

def plot_abs_and_relative(df_binned: pd.DataFrame, title_stem: str, stem: str):
    """Make both absolute and per-device relative heatmaps."""
    y_labels = make_y_labels(df_binned.index, BIN_SEC)
    X = list(df_binned.columns)
    M_abs = df_binned.values.astype(float)
    # Absolute (global scale)
    heatmap_cellwise(M_abs, X, y_labels,
                     f"{title_stem} — absolute",
                     OUTDIR / f"{stem}_absolute.png")
    # Per-device relative (min–max within each column)
    M_rel = per_device_minmax(M_abs)
    heatmap_cellwise(M_rel, X, y_labels,
                     f"{title_stem} — per-device RELATIVE (0–1)",
                     OUTDIR / f"{stem}_relative.png",
                     # use full 0–1 linear scale to emphasize within-device changes
                     norm=Normalize(vmin=0, vmax=1),
                     cbar_label="Relative level (0–1)")

def cluster_and_plot(df_binned: pd.DataFrame, title_stem: str, stem: str):
    """Cluster columns and plot absolute + relative versions."""
    y_labels = make_y_labels(df_binned.index, BIN_SEC)
    X = list(df_binned.columns)
    M_abs = df_binned.values.astype(float)

    # cluster order based on absolute (percentile/log) view
    norm_abs, _ = build_norm(M_abs)
    Z = linkage((M_abs + (LOG_EPS if isinstance(norm_abs, LogNorm) else 0.0)).T,
                method="average", metric="euclidean")
    order = leaves_list(Z)
    Xo = [X[i] for i in order]

    # absolute clustered
    heatmap_cellwise(M_abs[:, order], Xo, y_labels,
                     f"{title_stem} — absolute (columns clustered)",
                     OUTDIR / f"{stem}_clustered_absolute.png")

    # relative clustered
    M_rel = per_device_minmax(M_abs)[:, order]
    heatmap_cellwise(M_rel, Xo, y_labels,
                     f"{title_stem} — per-device RELATIVE (clustered)",
                     OUTDIR / f"{stem}_clustered_relative.png",
                     norm=Normalize(vmin=0, vmax=1),
                     cbar_label="Relative level (0–1)")


# Main
if __name__ == "__main__":
    df = load_and_clean(CSV_PATH)
    dev_cols = detect_device_columns(df)
    coerce_numeric(df, dev_cols)

    macros   = [c for c in dev_cols if re.match(r"^[ABCD]\d+$", c)]  # A1..D6
    subsites = [c for c in dev_cols if "_" in c]                     # A1_11..D6_44

    # 1) Macro A1–D6 (subsites collapsed)
    macro_map = {}
    for macro in macros:
        subs = [c for c in subsites if c.startswith(macro + "_")]
        cols = ([macro] if macro in df.columns else []) + subs
        if cols: macro_map[macro] = cols
    macro_df = pd.DataFrame({m: sum_numeric(df, cols) for m, cols in macro_map.items()})
    macro_binned = bin_and_sum(pd.concat([df, macro_df], axis=1), list(macro_df.columns), BIN_SEC)
    plot_abs_and_relative(macro_binned, "Macro electrodes (A1–D6, subsites collapsed)", "macro")

    # 2) Per-letter subsites
    for letter in ["A", "B", "C", "D"]:
        letter_cols = [c for c in dev_cols if c.startswith(letter)]
        if not letter_cols:
            continue
        letter_binned = bin_and_sum(df, letter_cols, BIN_SEC)
        plot_abs_and_relative(letter_binned, f"{letter}-group subsites", f"{letter}_subsites")

    # 3) All subsites clustered
    if subsites:
        subs_binned = bin_and_sum(df, subsites, BIN_SEC)
        cluster_and_plot(subs_binned, "All subsites (A1_11 … D6_44)", "all_subsites")

    print("Done.")
