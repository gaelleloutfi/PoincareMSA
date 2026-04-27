#!/usr/bin/env python3
"""
Phase 2: Analysis script for the Leave-One-Out Benchmark.
Generates plots, outlier analysis, and a markdown summary report
from the results generated in Phase 1.
"""

import argparse
import logging
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

# Metrics shared across all plot/stat functions
PLOT_METRICS: dict[str, tuple[str, str]] = {
    "delta_qlocal":   ("Delta Q_local",     "Delta Q_local"),
    "delta_qglobal":  ("Delta Q_global",    "Delta Q_global"),
    "insertion_time": ("Insertion Time (s)", "Insertion Time (s)"),
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze Leave-One-Out Benchmark Results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="benchmark_results",
        help="Directory containing per_iteration_results.csv",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="benchmark_results/figures",
        help="Directory where figures (.png) will be saved",
    )
    parser.add_argument(
        "--stats_dir",
        type=str,
        default="",
        help="Directory where CSV summaries will be saved. "
             "Defaults to --results_dir when left empty.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="",
        help="Comma-separated datasets to analyse. Empty = all found in CSV.",
    )
    return parser.parse_args()

def load_results(results_dir: str) -> pd.DataFrame:
    """
    Load the benchmark results CSV, validate columns, and prepare clean metrics.
    """
    csv_path = os.path.join(results_dir, "per_iteration_results.csv")
    if not os.path.isfile(csv_path):
        logger.error(f"Cannot find results CSV at: {csv_path}")
        sys.exit(1)
        
    logger.info(f"Loading results from: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        logger.error(f"Failed to read CSV {csv_path}: {e}")
        sys.exit(1)
        
    required_cols = [
        "dataset", "protein_id", "method", "insertion_time",
        "Qlocal_red", "Qlocal_after",
        "Qglobal_red", "Qglobal_after",
        "full_map_radius"
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.error(f"Missing required columns in CSV: {missing_cols}")
        sys.exit(1)
        
    # Calculate delta metrics
    df["delta_qlocal"] = df["Qlocal_after"] - df["Qlocal_red"]
    df["delta_qglobal"] = df["Qglobal_after"] - df["Qglobal_red"]
    
    logger.info(f"Successfully loaded {len(df)} rows across {df['dataset'].nunique()} dataset(s).")
    return df

def _plot_metric(df: pd.DataFrame, metric: str, title: str, ylabel: str, plot_type: str, output_dir: str, file_prefix: str):
    """
    Helper to generate a single plot for a given metric across methods.
    plot_type: 'boxplot' or 'violinplot'
    """
    plt.figure(figsize=(10, 6))
    if plot_type == 'boxplot':
        sns.boxplot(data=df, x="method", y=metric)
    elif plot_type == 'violinplot':
        sns.violinplot(data=df, x="method", y=metric)
        
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("Method")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{file_prefix}_{metric}.png"))
    plt.close()

def generate_boxplots(df: pd.DataFrame, output_dir: str):
    """
    Generate boxplots for delta_Qlocal, delta_Qglobal, and insertion_time.
    """
    logger.info("Generating boxplots...")
    for metric, (title, ylabel) in PLOT_METRICS.items():
        if metric not in df.columns:
            continue
        # Aggregated
        _plot_metric(df, metric, f"Aggregated {title} by Method",
                     ylabel, 'boxplot', output_dir, "boxplot_aggregated")
        # Per dataset
        for dataset in df["dataset"].unique():
            ds_df = df[df["dataset"] == dataset]
            _plot_metric(ds_df, metric, f"{dataset}: {title} by Method",
                         ylabel, 'boxplot', output_dir, f"boxplot_{dataset}")

def generate_violin_plots(df: pd.DataFrame, output_dir: str):
    """
    Generate violin plots for delta metrics and runtime.
    """
    logger.info("Generating violin plots...")
    for metric, (title, ylabel) in PLOT_METRICS.items():
        if metric not in df.columns:
            continue
        # Aggregated
        _plot_metric(df, metric, f"Aggregated {title} by Method",
                     ylabel, 'violinplot', output_dir, "violin_aggregated")
        # Per dataset
        for dataset in df["dataset"].unique():
            ds_df = df[df["dataset"] == dataset]
            _plot_metric(ds_df, metric, f"{dataset}: {title} by Method",
                         ylabel, 'violinplot', output_dir, f"violin_{dataset}")

def _plot_scatter(df: pd.DataFrame, x_metric: str, y_metric: str, title: str, xlabel: str, ylabel: str, output_dir: str, file_prefix: str):
    """
    Helper to generate a scatter plot colored by method.
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x=x_metric, y=y_metric, hue="method", alpha=0.7)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{file_prefix}_{y_metric}.png"))
    plt.close()

def generate_scatter_plots(df: pd.DataFrame, output_dir: str):
    """
    Generate scatter plots (radius vs delta metrics/runtime).
    """
    logger.info("Generating scatter plots...")
    if "full_map_radius" not in df.columns:
        logger.warning("full_map_radius not found, skipping scatter plots.")
        return
        
    for metric, (title, ylabel) in PLOT_METRICS.items():
        if metric not in df.columns:
            continue
        # Aggregated
        _plot_scatter(df, "full_map_radius", metric,
                      f"Aggregated: Radius vs {title}",
                      "Full Map Radius", ylabel,
                      output_dir, "scatter_aggregated")
        # Per dataset
        for dataset in df["dataset"].unique():
            ds_df = df[df["dataset"] == dataset]
            _plot_scatter(ds_df, "full_map_radius", metric,
                          f"{dataset}: Radius vs {title}",
                          "Full Map Radius", ylabel,
                          output_dir, f"scatter_{dataset}")

def compute_summary_statistics(df: pd.DataFrame, output_dir: str):
    """
    Compute per-method summary statistics for key metrics.
    """
    logger.info("Computing per-method summary statistics...")
    metrics = ["delta_qlocal", "delta_qglobal", "insertion_time"]
    
    available_metrics = [m for m in metrics if m in df.columns]
    
    if available_metrics:
        summary_df = df.groupby("method")[available_metrics].agg(["mean", "median", "std", "min", "max"])
        summary_df.columns = ["_".join(col).strip() for col in summary_df.columns.values]
        summary_df = summary_df.reset_index()
        
        out_path = os.path.join(output_dir, "summary_statistics.csv")
        summary_df.to_csv(out_path, index=False)
        logger.info(f"Saved summary statistics to {out_path}")
        return summary_df
    return None

def analyze_outliers(df: pd.DataFrame, output_dir: str):
    """
    Perform Tukey outlier analysis (1.5 IQR) on delta_Qlocal per method.
    Identifies outliers and computes per-method outlier overlap table.
    """
    logger.info("Analyzing outliers for delta_qlocal...")
    
    if "delta_qlocal" not in df.columns:
        logger.warning("delta_qlocal not found, skipping outlier analysis.")
        return
        
    outlier_df = df.copy()
    outlier_df["is_outlier"] = False
    
    for method in outlier_df["method"].unique():
        method_mask = outlier_df["method"] == method
        method_data = outlier_df.loc[method_mask, "delta_qlocal"]
        
        # Handle cases with very few points or NaNs
        if len(method_data.dropna()) < 4:
            continue
            
        q1 = method_data.quantile(0.25)
        q3 = method_data.quantile(0.75)
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outlier_mask = method_mask & ((outlier_df["delta_qlocal"] < lower_bound) | (outlier_df["delta_qlocal"] > upper_bound))
        outlier_df.loc[outlier_mask, "is_outlier"] = True
        
        logger.info(f"Method '{method}': Found {outlier_mask.sum()} outliers for delta_qlocal "
                    f"(bounds: [{lower_bound:.4f}, {upper_bound:.4f}])")

    outliers_path = os.path.join(output_dir, "outliers_flagged.csv")
    outlier_df.to_csv(outliers_path, index=False)
    
    # Outlier overlap table
    try:
        pivot_outliers = outlier_df.pivot_table(
            index=["dataset", "protein_id"], 
            columns="method", 
            values="is_outlier",
            aggfunc="first"
        ).fillna(False)
        
        methods = pivot_outliers.columns
        overlap_matrix = pd.DataFrame(index=methods, columns=methods, dtype=int)
        
        for m1 in methods:
            for m2 in methods:
                overlap_matrix.loc[m1, m2] = (pivot_outliers[m1] & pivot_outliers[m2]).sum()
                
        overlap_path = os.path.join(output_dir, "outlier_overlap_matrix.csv")
        overlap_matrix.to_csv(overlap_path)
        logger.info(f"Saved outlier overlap matrix to {overlap_path}")
    except Exception as e:
        logger.warning(f"Could not compute outlier overlap matrix: {e}")
        
    outliers_only = outlier_df[outlier_df["is_outlier"]]
    outliers_summary_path = os.path.join(output_dir, "outliers_summary.csv")
    outliers_only.to_csv(outliers_summary_path, index=False)
    logger.info(f"Saved {len(outliers_only)} identified outliers to {outliers_summary_path}")
    
    return outlier_df

def generate_outlier_plots(outlier_df: pd.DataFrame, output_dir: str):
    """
    Generate scatter plots highlighting outlier examples for delta_Qlocal per method.
    """
    logger.info("Generating outlier visualization plots...")
    if outlier_df is None or "is_outlier" not in outlier_df.columns or "full_map_radius" not in outlier_df.columns:
        logger.warning("Required columns for outlier plots not found, skipping.")
        return
        
    for method in outlier_df["method"].unique():
        method_df = outlier_df[outlier_df["method"] == method]
        if not method_df["is_outlier"].any():
            continue
            
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            data=method_df, 
            x="full_map_radius", 
            y="delta_qlocal", 
            hue="is_outlier", 
            palette={False: "gray", True: "red"},
            style="is_outlier",
            alpha=0.7
        )
        plt.title(f"Outliers for {method}: Radius vs Delta Q_local")
        plt.xlabel("Full Map Radius")
        plt.ylabel("Delta Q_local")
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, f"outlier_examples_{method}.png"))
        plt.close()

def analyze_radial_bins(df: pd.DataFrame, output_dir: str):
    """
    Bin by full_map_radius and compute mean/median metrics per bin per method.
    """
    logger.info("Analyzing radial bins...")
    if "full_map_radius" not in df.columns:
        logger.warning("full_map_radius not found, skipping radial bin analysis.")
        return
        
    df_bins = df.copy()
    try:
        df_bins["radius_bin"] = pd.qcut(df_bins["full_map_radius"], q=3, labels=["Center", "Mid", "Periphery"])
    except Exception as e:
        logger.warning(f"Failed to qcut radius bins: {e}. Falling back to 3 equal-width bins.")
        df_bins["radius_bin"] = pd.cut(df_bins["full_map_radius"], bins=3, labels=["Inner", "Mid", "Outer"])
        
    metrics = ["delta_qlocal", "delta_qglobal", "insertion_time"]
    available_metrics = [m for m in metrics if m in df_bins.columns]
    
    if available_metrics:
        bin_summary = df_bins.groupby(["method", "radius_bin"], observed=False)[available_metrics].agg(["mean", "median"])
        bin_summary.columns = ["_".join(col).strip() for col in bin_summary.columns.values]
        bin_summary = bin_summary.reset_index()
        
        bin_path = os.path.join(output_dir, "radial_bin_summary.csv")
        bin_summary.to_csv(bin_path, index=False)
        logger.info(f"Saved radial bin summary to {bin_path}")

def compute_correlations(df: pd.DataFrame, output_dir: str):
    """
    Compute Spearman correlations (e.g. radius vs delta metrics).
    """
    logger.info("Computing Spearman correlations...")
    if "full_map_radius" not in df.columns:
        logger.warning("full_map_radius not found, skipping correlation analysis.")
        return
        
    metrics = ["delta_qlocal", "delta_qglobal"]
    available_metrics = [m for m in metrics if m in df.columns]
    
    if not available_metrics:
        return
        
    results = []
    for method in df["method"].unique():
        method_df = df[df["method"] == method]
        row = {"method": method}
        
        for metric in available_metrics:
            try:
                corr = method_df[["full_map_radius", metric]].corr(method="spearman").iloc[0, 1]
                row[f"corr_radius_vs_{metric}"] = corr
            except Exception as e:
                logger.warning(f"Failed to compute correlation for {method} {metric}: {e}")
                row[f"corr_radius_vs_{metric}"] = None
                
        results.append(row)
        
    corr_df = pd.DataFrame(results)
    corr_path = os.path.join(output_dir, "spearman_correlations.csv")
    corr_df.to_csv(corr_path, index=False)
    logger.info(f"Saved Spearman correlations to {corr_path}")

def generate_markdown_report(df: pd.DataFrame, figures_dir: str, report_dir: str):
    """
    Generate a markdown summary report of the findings.
    """
    logger.info("Generating markdown report...")
    
    report_path = os.path.join(report_dir, "leave_one_out_benchmark_summary.md")
    
    datasets_analyzed = df["dataset"].unique()
    methods = df["method"].unique()
    num_rows = len(df)
    
    # Load CSVs
    try:
        summary_stats = pd.read_csv(os.path.join(figures_dir, "summary_statistics.csv")).to_markdown(index=False)
    except Exception:
        summary_stats = "*Data not available*"
        
    try:
        corr_stats = pd.read_csv(os.path.join(figures_dir, "spearman_correlations.csv")).to_markdown(index=False)
    except Exception:
        corr_stats = "*Data not available*"
        
    try:
        outlier_summary = pd.read_csv(os.path.join(figures_dir, "outliers_summary.csv"))
        num_outliers = len(outlier_summary)
    except Exception:
        num_outliers = 0
        
    try:
        bin_stats = pd.read_csv(os.path.join(figures_dir, "radial_bin_summary.csv")).to_markdown(index=False)
    except Exception:
        bin_stats = "*Data not available*"

    lines = [
        "# Leave-One-Out Benchmark Summary",
        "",
        "## Datasets and Scope",
        f"- **Datasets Analyzed**: {', '.join(datasets_analyzed)}",
        f"- **Total Rows**: {num_rows}",
        f"- **Methods Evaluated**: {', '.join(methods)}",
        "",
        "## Per-Method Performance Trends",
        "Summary statistics for `delta_qlocal`, `delta_qglobal`, and `insertion_time`.",
        "",
        summary_stats,
        "",
        "## Radial-Bin Trends",
        "Performance variations depending on the protein's radial position in the full map.",
        "",
        bin_stats,
        "",
        "## Correlation Results",
        "Spearman rank correlations between `full_map_radius` and delta metrics.",
        "",
        corr_stats,
        "",
        "## Outlier Summary",
        f"A total of **{num_outliers}** outliers were identified using the Tukey 1.5 IQR rule for `delta_qlocal`.",
        "Detailed overlapping matrices and raw outlier rows are available in the generated CSV files."
    ]
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        
    logger.info(f"Markdown report generated at: {report_path}")

def main():
    args = parse_args()

    # Resolve stats_dir (CSV summaries) — separate from figures output_dir
    stats_dir = args.stats_dir if args.stats_dir else args.results_dir

    # 1. Ensure output directories exist
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(stats_dir, exist_ok=True)
    logger.info("Figures dir : %s", args.output_dir)
    logger.info("Stats dir   : %s", stats_dir)

    # 2. Load and validate results
    df = load_results(args.results_dir)

    # 3. Optionally filter to requested datasets
    if args.datasets:
        datasets_to_keep = [d.strip() for d in args.datasets.split(",") if d.strip()]
        logger.info("Filtering datasets to: %s", datasets_to_keep)
        available = set(df["dataset"].unique())
        for d in datasets_to_keep:
            if d not in available:
                logger.warning("Requested dataset '%s' not found in results.", d)
        df = df[df["dataset"].isin(datasets_to_keep)].copy()
        if len(df) == 0:
            logger.error("No data remaining after dataset filtering.")
            sys.exit(1)
        logger.info("Rows after filtering: %d", len(df))

    # 4. Generate figures
    generate_boxplots(df, args.output_dir)
    generate_violin_plots(df, args.output_dir)
    generate_scatter_plots(df, args.output_dir)

    # 5. Compute statistics → written to stats_dir
    compute_summary_statistics(df, stats_dir)
    outlier_df = analyze_outliers(df, stats_dir)
    analyze_radial_bins(df, stats_dir)
    compute_correlations(df, stats_dir)

    # 6. Outlier visualisations (reads from stats_dir, writes to output_dir)
    if outlier_df is not None:
        generate_outlier_plots(outlier_df, args.output_dir)

    # 7. Markdown report — anchored to results_dir, not output_dir
    report_dir = os.path.join(args.results_dir, "reports")
    os.makedirs(report_dir, exist_ok=True)
    generate_markdown_report(df, stats_dir, report_dir)

    logger.info("Phase 2 analysis complete.")

if __name__ == "__main__":
    main()
