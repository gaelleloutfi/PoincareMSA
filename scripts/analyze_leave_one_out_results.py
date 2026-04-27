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
from pathlib import Path
import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Leave-One-Out Benchmark Results")
    
    parser.add_argument(
        "--results_dir",
        type=str,
        default="benchmark_results",
        help="Directory containing per_iteration_results.csv"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="benchmark_results/figures",
        help="Directory where figures will be saved"
    )
    
    parser.add_argument(
        "--datasets",
        type=str,
        default="",
        help="Comma-separated list of datasets to analyze. If empty, all found in CSV are used."
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
        "qlocal_reduced_before", "qlocal_after",
        "qglobal_reduced_before", "qglobal_after",
        "full_map_radius"
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.error(f"Missing required columns in CSV: {missing_cols}")
        sys.exit(1)
        
    # Calculate delta metrics
    df["delta_qlocal"] = df["qlocal_after"] - df["qlocal_reduced_before"]
    df["delta_qglobal"] = df["qglobal_after"] - df["qglobal_reduced_before"]
    
    logger.info(f"Successfully loaded {len(df)} rows across {df['dataset'].nunique()} dataset(s).")
    return df

def generate_boxplots(df, output_dir: str):
    """
    Generate boxplots for delta_Qlocal, delta_Qglobal, and insertion_time.
    TODO: Implement seaborn/matplotlib boxplots.
    """
    logger.info("Generating boxplots...")
    pass

def generate_violin_plots(df, output_dir: str):
    """
    Generate violin plots for delta metrics and runtime.
    TODO: Implement seaborn violin plots.
    """
    logger.info("Generating violin plots...")
    pass

def generate_scatter_plots(df, output_dir: str):
    """
    Generate scatter plots (e.g. radius vs delta metrics).
    TODO: Implement seaborn scatter plots.
    """
    logger.info("Generating scatter plots...")
    pass

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

def generate_markdown_report(df, output_dir: str):
    """
    Generate a markdown summary report of the findings.
    TODO: Create the markdown string and write to file.
    """
    logger.info("Generating markdown report...")
    pass

def main():
    args = parse_args()
    
    # 1. Setup output directory
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Output directory ready: {args.output_dir}")
    
    # 2. Load results
    df = load_results(args.results_dir)
    
    # 3. Filter by dataset if requested
    if args.datasets:
        datasets_to_keep = [d.strip() for d in args.datasets.split(",") if d.strip()]
        logger.info(f"Filtering datasets to: {datasets_to_keep}")
        
        # Validate that the requested datasets exist in the data
        available_datasets = set(df["dataset"].unique())
        for d in datasets_to_keep:
            if d not in available_datasets:
                logger.warning(f"Requested dataset '{d}' not found in the results.")
                
        df = df[df["dataset"].isin(datasets_to_keep)].copy()
        
        if len(df) == 0:
            logger.error("No data remaining after dataset filtering.")
            sys.exit(1)
            
        logger.info(f"Data remaining after filtering: {len(df)} rows.")
    
    # 4. Generate plots and analyses
    # generate_boxplots(df, args.output_dir)
    # generate_violin_plots(df, args.output_dir)
    # generate_scatter_plots(df, args.output_dir)
    compute_summary_statistics(df, args.output_dir)
    analyze_outliers(df, args.output_dir)
    analyze_radial_bins(df, args.output_dir)
    compute_correlations(df, args.output_dir)
    
    # 5. Generate final report
    # report_dir = os.path.join(os.path.dirname(args.output_dir), "reports")
    # os.makedirs(report_dir, exist_ok=True)
    # generate_markdown_report(df, report_dir)
    
    logger.info("Phase 2 Analysis skeleton complete.")

if __name__ == "__main__":
    main()
