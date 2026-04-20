#!/usr/bin/env python3
"""
Leave-One-Out Benchmark for Poincaré Map Point Insertion — Phase 1
====================================================================

Protocol
--------
For each dataset (globins, thioredoxins, kinases):
  1. Load all features and build the FULL Poincaré map (reference).
  2. Compute full-map Qlocal, Qglobal, radial coordinates.
  3. Sample a subset of proteins to remove (stratified across radial bins).
  4. For each removed protein:
       a. Build a REDUCED map from the remaining N-1 proteins.
       b. Compute reduced-map Qlocal / Qglobal BEFORE insertion.
       c. Insert the removed protein with Methods 1, 2, 3.
       d. After each insertion compute Qlocal / Qglobal AFTER insertion,
          plus neighbour-overlap metrics and timing.
  5. Save structured CSV results and checkpoints.

Primary metrics
---------------
  - Qlocal_reduced_before / Qglobal_reduced_before
  - Qlocal_after / Qglobal_after per method
  - delta_Qlocal = Qlocal_after - Qlocal_reduced_before  (idem for Qglobal)
  - insertion_time_sec
  - map_build_time_sec

Coordinate distance to the original full-map position is stored as a
*reference-only* field and is NOT used as a primary benchmark metric,
because rebuilding with N-1 points produces a genuinely different projection.

Usage
-----
  python scripts/run_leave_one_out_benchmark.py \\
      --dataset globins \\
      --n_remove 25 \\
      --output_dir benchmark_results \\
      --seed 42

Run with --help for all options.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup — make project root importable regardless of CWD
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent          # scripts/
_PROJECT_ROOT = _SCRIPT_DIR.parent                     # PoincareMSA/
_BUILD_PKG = _PROJECT_ROOT / "scripts" / "build_poincare_map"

for _p in [str(_PROJECT_ROOT), str(_BUILD_PKG)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("loo_benchmark")


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------
# Each entry maps a dataset name to the information needed to load it.
# 'input_type' is either 'pssm' (folder of .aamtx files) or 'plm' (folder
# of .pt embedding files).
DATASET_REGISTRY: dict[str, dict[str, Any]] = {
    "globins": {
        "input_type": "pssm",
        "data_path": "examples/globins/globins_data/fasta0.9",
        "annotation_path": "examples/globins/globin_colors_new.csv",
        "annotation_id_col": None,   # will be inferred
        # Map-building hyperparameters matching Breton's setup
        "distlocal": "cosine",
        "knn": 5,
        "sigma": 1.0,
        "gamma": 2.0,
        "distfn": "MFIsym",
        "lossfn": "klSym",
        "distr": "laplace",
    },
    "thioredoxins": {
        "input_type": "plm",
        "data_path": "embeddings/ankh_base_thioredoxins",
        "annotation_path": "examples/thioredoxins/thioredoxin_annotation.csv",
        "annotation_id_col": "proteins_id",
        "distlocal": "cosine",
        "knn": 5,
        "sigma": 1.0,
        "gamma": 2.0,
        "distfn": "MFIsym",
        "lossfn": "klSym",
        "distr": "laplace",
    },
    "kinases": {
        "input_type": "plm",
        "data_path": "embeddings/ankh_base_kinases",
        "annotation_path": "examples/kinases/kinase_group_new.csv",
        "annotation_id_col": "proteins_id",
        "distlocal": "cosine",
        "knn": 5,
        "sigma": 1.0,
        "gamma": 2.0,
        "distfn": "MFIsym",
        "lossfn": "klSym",
        "distr": "laplace",
    },
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_leave_one_out_benchmark.py",
        description=(
            "Leave-one-out benchmark for Poincaré map point insertion.\n"
            "Builds a reduced map for each removed protein and evaluates "
            "Methods 1, 2, and 3."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- Dataset -------------------------------------------------------
    ds_group = p.add_argument_group("Dataset")
    ds_group.add_argument(
        "--dataset",
        choices=list(DATASET_REGISTRY),
        required=True,
        help="Protein family to benchmark.",
    )
    ds_group.add_argument(
        "--data_path",
        default=None,
        help=(
            "Override the default data path for the dataset "
            "(folder of .aamtx or .pt files)."
        ),
    )
    ds_group.add_argument(
        "--annotation_path",
        default=None,
        help="Override the default annotation CSV path.",
    )

    # ---- Sampling ------------------------------------------------------
    sample_group = p.add_argument_group("Sampling")
    sample_group.add_argument(
        "--n_remove",
        type=int,
        default=25,
        help=(
            "Number of proteins to remove and reinsert. "
            "Use a small value (e.g. 5) for a quick sanity check. "
            "Use 0 to remove ALL proteins (full leave-one-out)."
        ),
    )
    sample_group.add_argument(
        "--sample_mode",
        choices=["stratified_radius", "random"],
        default="stratified_radius",
        help=(
            "How to select proteins to remove. "
            "'stratified_radius' spreads selections across center / mid / "
            "periphery radial bins; 'random' draws uniformly."
        ),
    )
    sample_group.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global random seed for reproducibility.",
    )

    # ---- Map-building hyperparameters ----------------------------------
    map_group = p.add_argument_group("Map hyperparameters")
    map_group.add_argument(
        "--epochs",
        type=int,
        default=300,
        help="Number of training epochs for each Poincaré map.",
    )
    map_group.add_argument(
        "--knn",
        type=int,
        default=None,
        help=(
            "K for the KNN graph. "
            "If not set, uses the dataset default from the registry."
        ),
    )
    map_group.add_argument(
        "--sigma",
        type=float,
        default=None,
        help="RFA bandwidth sigma. If not set, uses the dataset default.",
    )
    map_group.add_argument(
        "--gamma",
        type=float,
        default=None,
        help="Embedding bandwidth gamma. If not set, uses the dataset default.",
    )
    map_group.add_argument(
        "--lr",
        type=float,
        default=0.1,
        help="Learning rate for the Riemannian SGD optimiser.",
    )
    map_group.add_argument(
        "--dim",
        type=int,
        default=2,
        help="Embedding dimensionality (should be 2 for Poincaré disc).",
    )
    map_group.add_argument(
        "--earlystop",
        type=float,
        default=0.0,
        help="Early-stop threshold for epoch-to-epoch loss delta (0 = disabled).",
    )
    map_group.add_argument(
        "--connected",
        type=int,
        choices=[0, 1],
        default=1,
        help="Force the KNN graph to be connected (1=yes, 0=no).",
    )

    # ---- Insertion hyperparameters ------------------------------------
    ins_group = p.add_argument_group("Insertion hyperparameters")
    ins_group.add_argument(
        "--n_steps_insert",
        type=int,
        default=500,
        help="Gradient steps for Methods 2 and 3 (infer_embedding_for_point).",
    )
    ins_group.add_argument(
        "--lr_insert",
        type=float,
        default=0.05,
        help="Learning rate for Methods 2 and 3.",
    )
    ins_group.add_argument(
        "--k_quality",
        type=int,
        default=5,
        help="Number of neighbours k used when computing Qlocal and Qglobal.",
    )
    ins_group.add_argument(
        "--neighbor_overlap_k",
        type=int,
        nargs="+",
        default=[5, 10],
        help="k values for neighbour-overlap metric, e.g. --neighbor_overlap_k 5 10.",
    )

    # ---- Output --------------------------------------------------------
    out_group = p.add_argument_group("Output")
    out_group.add_argument(
        "--output_dir",
        default="benchmark_results",
        help="Root directory for all output files.",
    )
    out_group.add_argument(
        "--resume",
        action="store_true",
        help=(
            "If a partial results file already exists in output_dir, "
            "skip already-completed (dataset, protein_id, method) triples."
        ),
    )
    out_group.add_argument(
        "--save_embeddings",
        action="store_true",
        help="Save per-iteration reduced and inserted embeddings to disk.",
    )

    # ---- Verbosity / debug --------------------------------------------
    p.add_argument(
        "--debug",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )

    return p


def resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    """
    Merge CLI arguments with dataset registry defaults.

    Fields that are None fall back to the per-dataset registry value.
    Relative paths are resolved against the project root.
    """
    registry = DATASET_REGISTRY[args.dataset]

    # Fill in dataset-level defaults for hyperparameters the user did not set
    for field in ("knn", "sigma", "gamma"):
        if getattr(args, field) is None:
            setattr(args, field, registry[field])

    # Copy fixed registry fields onto args for easy downstream access
    for field in ("input_type", "distlocal", "distfn", "lossfn", "distr"):
        setattr(args, field, registry[field])

    # Resolve paths
    root = _PROJECT_ROOT
    data_path = args.data_path or registry["data_path"]
    annotation_path = args.annotation_path or registry.get("annotation_path")

    args.data_path = str(root / data_path)
    args.annotation_path = str(root / annotation_path) if annotation_path else None
    args.annotation_id_col = registry.get("annotation_id_col")

    # Output directory (absolute)
    args.output_dir = str(Path(args.output_dir).resolve())

    return args


# ---------------------------------------------------------------------------
# Output directory helpers
# ---------------------------------------------------------------------------

def setup_output_dir(output_dir: str, dataset: str) -> dict[str, str]:
    """
    Create the output directory tree and return a dict of important paths.
    """
    base = Path(output_dir)
    paths = {
        "base":            str(base),
        "figures":         str(base / "figures"),
        "reports":         str(base / "reports"),
        "partial_results": str(base / f"per_iteration_results_{dataset}_partial.csv"),
        "final_results":   str(base / "per_iteration_results.csv"),
        "errors":          str(base / f"errors_{dataset}.jsonl"),
        "full_map_emb":    str(base / f"full_map_{dataset}.csv"),
        "full_map_meta":   str(base / f"full_map_{dataset}_meta.json"),
    }
    for d in (paths["base"], paths["figures"], paths["reports"]):
        os.makedirs(d, exist_ok=True)
    return paths


# ---------------------------------------------------------------------------
# Stub functions (to be implemented in subsequent steps)
# ---------------------------------------------------------------------------

def load_dataset(args: argparse.Namespace):
    """
    Load features and labels for the selected dataset.

    Returns
    -------
    features : np.ndarray, shape (N, d)
    labels   : np.ndarray of str, shape (N,)
    """
    raise NotImplementedError("load_dataset — to be implemented")


def build_poincare_map(features: np.ndarray, args: argparse.Namespace, tmp_dir: str):
    """
    Build a Poincaré map from a feature matrix.

    Returns
    -------
    embeddings   : np.ndarray, shape (N, 2)
    loss         : float
    build_time   : float  (seconds, wall-clock)
    """
    raise NotImplementedError("build_poincare_map — to be implemented")


def compute_quality(features: np.ndarray, embeddings: np.ndarray, args: argparse.Namespace):
    """
    Compute Qlocal and Qglobal for a given (features, embeddings) pair.

    Returns
    -------
    qlocal  : float
    qglobal : float
    """
    raise NotImplementedError("compute_quality — to be implemented")


def sample_proteins_to_remove(
    embeddings_full: np.ndarray,
    labels: np.ndarray,
    n_remove: int,
    mode: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Select indices of proteins to remove, with an optional stratified-radius
    strategy so that center, intermediate, and peripheral proteins are
    represented.

    Returns
    -------
    indices : np.ndarray of int, shape (n_remove,)
    """
    raise NotImplementedError("sample_proteins_to_remove — to be implemented")


def compute_target_vector(
    removed_feat: np.ndarray,
    remaining_feats: np.ndarray,
    gamma: float,
    distlocal: str,
) -> np.ndarray:
    """
    Compute the target RFA-like probability vector for the removed protein
    w.r.t. the remaining proteins' feature space.

    Returns
    -------
    target : np.ndarray, shape (N-1,)
    """
    raise NotImplementedError("compute_target_vector — to be implemented")


def insert_method1_bary(model, target: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Method 1: pure hyperbolic barycenter (no gradient optimisation).

    Returns
    -------
    embedding   : np.ndarray, shape (dim,)
    elapsed_sec : float
    """
    raise NotImplementedError("insert_method1_bary — to be implemented")


def insert_method2_rand(
    model, target: np.ndarray, args: argparse.Namespace
) -> tuple[np.ndarray, float]:
    """
    Method 2: infer_embedding_for_point with random initialisation.

    Returns
    -------
    embedding   : np.ndarray, shape (dim,)
    elapsed_sec : float
    """
    raise NotImplementedError("insert_method2_rand — to be implemented")


def insert_method3_bary(
    model, target: np.ndarray, args: argparse.Namespace
) -> tuple[np.ndarray, float]:
    """
    Method 3: infer_embedding_for_point with barycenter initialisation.

    Returns
    -------
    embedding   : np.ndarray, shape (dim,)
    elapsed_sec : float
    """
    raise NotImplementedError("insert_method3_bary — to be implemented")


def compute_neighbor_overlap(
    full_emb: np.ndarray,
    removed_idx: int,
    inserted_emb: np.ndarray,
    reduced_emb: np.ndarray,
    k: int,
) -> float:
    """
    Compare the k nearest neighbours of the removed protein in the full map
    vs. the k nearest neighbours of the inserted point in the reduced+inserted map.

    Returns the fraction of shared neighbours (Jaccard-style overlap ∈ [0, 1]).
    """
    raise NotImplementedError("compute_neighbor_overlap — to be implemented")


def load_annotations(annotation_path: str, id_col: str | None) -> pd.DataFrame | None:
    """
    Load an annotation CSV and return it as a DataFrame, or None if the path
    is not set / does not exist.
    """
    raise NotImplementedError("load_annotations — to be implemented")


def build_result_row(
    dataset: str,
    protein_id: str,
    method: str,
    sample_mode: str,
    map_build_time: float,
    insertion_time: float,
    qlocal_reduced_before: float,
    qglobal_reduced_before: float,
    qlocal_after: float,
    qglobal_after: float,
    full_map_radius: float,
    inserted_radius: float,
    neighbor_overlaps: dict[int, float],
    local_density_proxy: float,
    annotation_row: dict | None,
) -> dict:
    """
    Assemble a single result row (dict) with all benchmark fields.
    """
    raise NotImplementedError("build_result_row — to be implemented")


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def append_row_to_partial(row: dict, partial_path: str) -> None:
    """Append a single result row to the partial CSV checkpoint file."""
    raise NotImplementedError("append_row_to_partial — to be implemented")


def log_error(error_path: str, dataset: str, protein_id: str, method: str, exc: Exception) -> None:
    """Record a failed iteration to the JSONL error log."""
    record = {
        "dataset": dataset,
        "protein_id": protein_id,
        "method": method,
        "error": type(exc).__name__,
        "message": str(exc),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(error_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def save_summary_tables(results_df: pd.DataFrame, output_dir: str) -> None:
    """
    Generate and save summary_by_method.csv and summary_by_radial_bin.csv
    from the final results DataFrame.
    """
    raise NotImplementedError("save_summary_tables — to be implemented")


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def run_benchmark(args: argparse.Namespace) -> None:
    """
    Top-level orchestration: calls all steps in sequence and handles errors.
    """
    paths = setup_output_dir(args.output_dir, args.dataset)
    logger.info("Output directory: %s", paths["base"])
    logger.info("Dataset:          %s  (input_type=%s)", args.dataset, args.input_type)
    logger.info("n_remove:         %d  (mode=%s, seed=%d)", args.n_remove, args.sample_mode, args.seed)

    rng = np.random.default_rng(args.seed)

    # ------------------------------------------------------------------
    # Step 1: Load features + labels
    # ------------------------------------------------------------------
    logger.info("[1/5] Loading features …")
    features, labels = load_dataset(args)
    N = features.shape[0]
    logger.info("      Loaded %d proteins, feature dim=%d", N, features.shape[1])

    # ------------------------------------------------------------------
    # Step 2: Build full map
    # ------------------------------------------------------------------
    logger.info("[2/5] Building full Poincaré map (%d proteins) …", N)
    tmp_full = os.path.join(paths["base"], "_tmp_full")
    os.makedirs(tmp_full, exist_ok=True)

    emb_full, loss_full, build_time_full = build_poincare_map(features, args, tmp_full)
    qlocal_full, qglobal_full = compute_quality(features, emb_full, args)

    # Radial coordinates (L2 norm of 2-D embedding)
    radii_full = np.linalg.norm(emb_full, axis=1)

    logger.info(
        "      Full map — loss=%.4e  build_time=%.1fs  "
        "Qlocal=%.4f  Qglobal=%.4f",
        loss_full, build_time_full, qlocal_full, qglobal_full,
    )

    # Persist full-map embeddings
    df_full_emb = pd.DataFrame(emb_full, columns=["pm1", "pm2"])
    df_full_emb["proteins_id"] = labels
    df_full_emb["radius"] = radii_full
    df_full_emb.to_csv(paths["full_map_emb"], index=False)

    full_map_meta = {
        "dataset": args.dataset,
        "N": int(N),
        "feature_dim": int(features.shape[1]),
        "qlocal_full": float(qlocal_full),
        "qglobal_full": float(qglobal_full),
        "loss_full": float(loss_full),
        "build_time_sec": float(build_time_full),
    }
    with open(paths["full_map_meta"], "w") as fh:
        json.dump(full_map_meta, fh, indent=2)

    # ------------------------------------------------------------------
    # Step 3: Sample proteins to remove
    # ------------------------------------------------------------------
    logger.info("[3/5] Sampling proteins to remove …")
    remove_indices = sample_proteins_to_remove(
        emb_full, labels, args.n_remove, args.sample_mode, rng
    )
    logger.info("      Selected %d proteins", len(remove_indices))

    # ------------------------------------------------------------------
    # Step 4: Load annotations (optional)
    # ------------------------------------------------------------------
    annotations_df = load_annotations(args.annotation_path, args.annotation_id_col)
    if annotations_df is not None:
        logger.info("      Loaded annotations (%d rows)", len(annotations_df))

    # ------------------------------------------------------------------
    # Step 5: Leave-one-out loop
    # ------------------------------------------------------------------
    logger.info("[4/5] Starting leave-one-out iterations …")
    all_rows: list[dict] = []
    n_ok = 0
    n_err = 0

    for iter_idx, remove_idx in enumerate(remove_indices):
        protein_id = str(labels[remove_idx])
        logger.info(
            "  [iter %d/%d]  protein=%s  radius=%.4f",
            iter_idx + 1, len(remove_indices), protein_id, radii_full[remove_idx],
        )

        # --- a) Build reduced map (N-1 proteins) ----------------------
        mask = np.ones(N, dtype=bool)
        mask[remove_idx] = False
        feats_reduced = features[mask]
        labels_reduced = labels[mask]

        try:
            tmp_red = os.path.join(paths["base"], f"_tmp_reduced_{iter_idx:04d}")
            os.makedirs(tmp_red, exist_ok=True)

            map_build_start = time.perf_counter()
            emb_red, loss_red, _ = build_poincare_map(feats_reduced, args, tmp_red)
            map_build_time = time.perf_counter() - map_build_start

            qlocal_red, qglobal_red = compute_quality(feats_reduced, emb_red, args)

        except Exception as exc:
            logger.warning("    Reduced map FAILED for %s: %s", protein_id, exc)
            log_error(paths["errors"], args.dataset, protein_id, "build_reduced_map", exc)
            n_err += 1
            continue

        # --- b) Compute insertion target vector -----------------------
        removed_feat = features[remove_idx]
        target = compute_target_vector(
            removed_feat, feats_reduced, args.gamma, args.distlocal
        )

        # --- c) Instantiate model loaded with reduced embeddings ------
        # (stub: model construction deferred to implementation step)
        model = None   # placeholder

        # --- d-f) Run each insertion method ---------------------------
        methods = {
            "bary_init":  lambda: insert_method1_bary(model, target),
            "infer_rand": lambda: insert_method2_rand(model, target, args),
            "infer_bary": lambda: insert_method3_bary(model, target, args),
        }

        for method_name, method_fn in methods.items():
            try:
                new_emb, insertion_time = method_fn()

                # --- g) Q-metrics after insertion ---------------------
                # Reconstruct combined embedding: N-1 reduced + 1 inserted
                # (position of the removed protein inserted at remove_idx)
                emb_after = np.insert(emb_red, remove_idx, new_emb.reshape(1, -1), axis=0)
                qlocal_after, qglobal_after = compute_quality(features, emb_after, args)

                # --- h) Neighbour overlap & density proxy -------------
                inserted_radius = float(np.linalg.norm(new_emb))
                neighbor_overlaps: dict[int, float] = {}
                for k in args.neighbor_overlap_k:
                    neighbor_overlaps[k] = compute_neighbor_overlap(
                        emb_full, remove_idx, new_emb, emb_red, k
                    )

                # Local density proxy: mean hyperbolic distance to top-5
                # neighbours in the full map (stub returns NaN for now)
                local_density_proxy = float("nan")   # placeholder

                # Annotation fields
                annotation_row: dict | None = None
                if annotations_df is not None:
                    # look up by protein_id — deferred to implementation step
                    pass

                row = build_result_row(
                    dataset=args.dataset,
                    protein_id=protein_id,
                    method=method_name,
                    sample_mode=args.sample_mode,
                    map_build_time=map_build_time,
                    insertion_time=insertion_time,
                    qlocal_reduced_before=qlocal_red,
                    qglobal_reduced_before=qglobal_red,
                    qlocal_after=qlocal_after,
                    qglobal_after=qglobal_after,
                    full_map_radius=float(radii_full[remove_idx]),
                    inserted_radius=inserted_radius,
                    neighbor_overlaps=neighbor_overlaps,
                    local_density_proxy=local_density_proxy,
                    annotation_row=annotation_row,
                )
                all_rows.append(row)
                append_row_to_partial(row, paths["partial_results"])
                n_ok += 1
                logger.info(
                    "    OK  %s  dQlocal=%+.4f  dQglobal=%+.4f  t=%.2fs",
                    method_name,
                    qlocal_after - qlocal_red,
                    qglobal_after - qglobal_red,
                    insertion_time,
                )

            except Exception as exc:
                logger.warning("    FAIL %s: %s", method_name, exc)
                log_error(paths["errors"], args.dataset, protein_id, method_name, exc)
                n_err += 1

    logger.info("[4/5] Loop complete — %d successes, %d failures.", n_ok, n_err)

    # ------------------------------------------------------------------
    # Step 6: Save final tables
    # ------------------------------------------------------------------
    logger.info("[5/5] Saving results …")
    if all_rows:
        results_df = pd.DataFrame(all_rows)
        results_df.to_csv(paths["final_results"], index=False)
        logger.info("      Saved %d rows → %s", len(results_df), paths["final_results"])
        save_summary_tables(results_df, paths["base"])
    else:
        logger.warning("      No successful rows to save.")

    logger.info("Done.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    args = resolve_args(args)
    run_benchmark(args)


if __name__ == "__main__":
    main()
