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
# Data-loading layer
# ---------------------------------------------------------------------------
# Public entry point : load_dataset(args)  ->  DatasetBundle
# The bundle carries everything later pipeline steps need.
# ─────────────────────────────────────────────────────────────────────────

from dataclasses import dataclass, field


@dataclass
class DatasetBundle:
    """
    Structured container for one loaded protein-family dataset.

    Attributes
    ----------
    features : np.ndarray, shape (N, d)
        Numeric feature matrix.  Each row is one protein.
        For PSSM datasets this is the flattened PSSM matrix;
        for PLM datasets this is the mean-pooled language-model embedding.
    labels : np.ndarray[str], shape (N,)
        Protein identifiers (file stem, e.g. "42") in the same row
        order as `features`.
    input_type : str
        Either ``'pssm'`` or ``'plm'``, matching DATASET_REGISTRY.
    data_path : str
        Resolved absolute path to the data folder that was loaded.
    annotations : pd.DataFrame | None
        Optional annotation table, indexed (or join-able) on the
        column given by `annotation_id_col`.  ``None`` if no
        annotation file was found / specified.
    annotation_id_col : str | None
        Column name in `annotations` that contains the protein identifier
        (same strings as `labels`).
    """
    features: np.ndarray
    labels: np.ndarray
    input_type: str
    data_path: str
    annotations: "pd.DataFrame | None" = field(default=None)
    annotation_id_col: "str | None" = field(default=None)

    # ------------------------------------------------------------------ #
    # Convenience helpers                                                   #
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return len(self.labels)

    def feature_dim(self) -> int:
        """Dimensionality of each feature vector."""
        return self.features.shape[1]

    def annotation_for(self, protein_id: str) -> "dict | None":
        """
        Return the annotation row for *protein_id* as a plain dict, or
        ``None`` if annotations are not loaded or the id is not found.

        The lookup is done on ``annotation_id_col`` (string comparison).
        If ``annotation_id_col`` is ``None``, the DataFrame index is used.
        """
        return lookup_annotation(self.annotations, self.annotation_id_col, protein_id)


# ─── Internal helpers ────────────────────────────────────────────────────

def _validate_data_path(data_path: str, dataset_name: str) -> None:
    """
    Raise a clear ``FileNotFoundError`` if *data_path* does not exist or is
    not a directory.
    """
    p = Path(data_path)
    if not p.exists():
        raise FileNotFoundError(
            f"[{dataset_name}] Data directory not found: {data_path}\n"
            "Check DATASET_REGISTRY or override with --data_path."
        )
    if not p.is_dir():
        raise NotADirectoryError(
            f"[{dataset_name}] Expected a directory but got a file: {data_path}"
        )


def _validate_features_and_labels(
    features: np.ndarray,
    labels: np.ndarray,
    dataset_name: str,
    expected_ext: str,
) -> None:
    """
    Run basic sanity checks on the loaded (features, labels) pair and raise
    informative errors when something looks wrong.

    Checks performed
    ----------------
    - At least one protein was loaded.
    - ``features`` is 2-D and ``labels`` is 1-D.
    - Row counts match.
    - No NaN / Inf values in features.
    - All feature vectors have non-zero variance (warn but do not raise).
    """
    n_feats, n_labels = len(features), len(labels)

    if n_feats == 0:
        raise ValueError(
            f"[{dataset_name}] No proteins were loaded from the data directory.\n"
            f"Expected files with extension '{expected_ext}'.  "
            "Check the path and the DATASET_REGISTRY."
        )
    if features.ndim != 2:
        raise ValueError(
            f"[{dataset_name}] Feature matrix must be 2-D, got shape {features.shape}."
        )
    if labels.ndim != 1:
        raise ValueError(
            f"[{dataset_name}] Labels array must be 1-D, got shape {labels.shape}."
        )
    if n_feats != n_labels:
        raise ValueError(
            f"[{dataset_name}] Mismatch: {n_feats} feature rows vs {n_labels} labels."
        )

    n_nan = int(np.isnan(features).sum())
    n_inf = int(np.isinf(features).sum())
    if n_nan > 0 or n_inf > 0:
        raise ValueError(
            f"[{dataset_name}] Feature matrix contains {n_nan} NaN(s) and "
            f"{n_inf} Inf value(s).  Check the source files."
        )

    # Warn about zero-variance dimensions (won't crash but may affect quality)
    zero_var_dims = int((np.std(features, axis=0) == 0).sum())
    if zero_var_dims > 0:
        logger.warning(
            "[%s] %d / %d feature dimensions have zero variance.",
            dataset_name, zero_var_dims, features.shape[1],
        )


def _load_pssm_features(data_path: str, dataset_name: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load PSSM features from a directory of ``.aamtx`` files.

    Delegates to ``prepare_data()`` from
    ``scripts/build_poincare_map/data.py``, which iterates over every
    ``.aamtx`` file, flattens each one to a 1-D vector, and stacks them
    into a feature matrix.

    Parameters
    ----------
    data_path : str
        Absolute path to the folder containing ``.aamtx`` files.
        A ``0.aamtx`` file is treated as the phylogenetic root and is
        *excluded* here (``withroot=False``) so that integer-named proteins
        start from ``1``.  This matches how the rest of the pipeline works.
    dataset_name : str
        Used only in error messages.

    Returns
    -------
    features : np.ndarray, shape (N, d)
    labels   : np.ndarray[str], shape (N,)
        Labels are the file stem (e.g. ``"42"`` for ``42.aamtx``).
    """
    try:
        from data import prepare_data  # noqa: PLC0415  (local import OK here)
    except ImportError as exc:
        raise ImportError(
            "Cannot import 'prepare_data' from scripts/build_poincare_map/data.py.\n"
            "Make sure the project root and build_poincare_map are on sys.path."
        ) from exc

    _validate_data_path(data_path, dataset_name)

    # Count .aamtx files so we can give a helpful error if there are none
    aamtx_files = [f for f in os.listdir(data_path) if f.endswith(".aamtx")]
    if not aamtx_files:
        raise FileNotFoundError(
            f"[{dataset_name}] No .aamtx files found in: {data_path}\n"
            "This dataset requires PSSM files produced by the prepare_data pipeline."
        )

    logger.debug(
        "[%s] Found %d .aamtx files in %s",
        dataset_name, len(aamtx_files), data_path,
    )

    # withroot=False: exclude the artificial root protein (0.aamtx)
    features_tensor, labels = prepare_data(data_path, withroot=False, fmt=".aamtx")

    features = features_tensor.numpy().astype(np.float64)
    labels = np.array([str(lbl) for lbl in labels])

    _validate_features_and_labels(features, labels, dataset_name, ".aamtx")
    logger.info(
        "[%s] Loaded PSSM features: %d proteins x %d dims  (from %s)",
        dataset_name, features.shape[0], features.shape[1], data_path,
    )
    return features, labels


def _load_plm_features(data_path: str, dataset_name: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load protein language-model (PLM) features from a directory of ``.pt``
    files.

    Delegates to ``prepare_embedding_data()`` from
    ``scripts/build_poincare_map/data.py``, which loads each ``.pt`` file
    (a dict with an ``'embedding'`` key), mean-pools over the sequence
    length dimension if necessary, and stacks the result into a matrix.

    Parameters
    ----------
    data_path : str
        Absolute path to the folder containing ``.pt`` embedding files.
    dataset_name : str
        Used only in error messages.

    Returns
    -------
    features : np.ndarray, shape (N, d)
    labels   : np.ndarray[str], shape (N,)
        Labels are the file stem (e.g. ``"42"`` for ``42.pt``).
    """
    try:
        from data import prepare_embedding_data  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "Cannot import 'prepare_embedding_data' from "
            "scripts/build_poincare_map/data.py."
        ) from exc

    _validate_data_path(data_path, dataset_name)

    pt_files = [f for f in os.listdir(data_path) if f.endswith(".pt")]
    if not pt_files:
        raise FileNotFoundError(
            f"[{dataset_name}] No .pt files found in: {data_path}\n"
            "This dataset requires PLM embedding files "
            "(see embeddings/ankh_base_*)."
        )

    logger.debug(
        "[%s] Found %d .pt files in %s",
        dataset_name, len(pt_files), data_path,
    )

    # withroot is unused for PLM datasets (no root convention)
    features_tensor, labels = prepare_embedding_data(data_path, withroot=True, fmt=".pt")

    features = features_tensor.numpy().astype(np.float64)
    labels = np.array([str(lbl) for lbl in labels])

    _validate_features_and_labels(features, labels, dataset_name, ".pt")
    logger.info(
        "[%s] Loaded PLM features: %d proteins x %d dims  (from %s)",
        dataset_name, features.shape[0], features.shape[1], data_path,
    )
    return features, labels


# ─── Annotation helpers ──────────────────────────────────────────────────

def load_annotations(annotation_path: "str | None", id_col: "str | None") -> "pd.DataFrame | None":
    """
    Load an annotation CSV and return it as a DataFrame.

    Returns ``None`` if *annotation_path* is ``None`` or the file does not
    exist (a warning is logged in the latter case so the user notices a
    misconfigured path).

    The ``id_col`` column is coerced to ``str`` so that it can be matched
    against the string labels returned by the feature loaders.

    Parameters
    ----------
    annotation_path : str | None
        Absolute path to the annotation CSV file.
    id_col : str | None
        Column in the CSV that holds protein identifiers.  If ``None``,
        the lookup falls back to the DataFrame integer index.

    Returns
    -------
    pd.DataFrame | None
    """
    if annotation_path is None:
        return None

    path = Path(annotation_path)
    if not path.exists():
        logger.warning(
            "Annotation file not found (skipping): %s", annotation_path
        )
        return None
    if not path.suffix.lower() == ".csv":
        logger.warning(
            "Annotation file does not look like a CSV: %s  (will try anyway)",
            annotation_path,
        )

    try:
        df = pd.read_csv(annotation_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Failed to read annotation file %s: %s", annotation_path, exc
        )
        return None

    if df.empty:
        logger.warning("Annotation file is empty: %s", annotation_path)
        return None

    # Coerce the identifier column to str for consistent matching
    if id_col is not None:
        if id_col not in df.columns:
            logger.warning(
                "Annotation id column '%s' not found in %s.\n"
                "Available columns: %s",
                id_col, annotation_path, list(df.columns),
            )
            # Return the raw dataframe anyway; annotation_for() will return None
        else:
            df[id_col] = df[id_col].astype(str)

    logger.info(
        "Loaded annotations: %d rows, %d columns  (id_col='%s')  from %s",
        len(df), len(df.columns), id_col, annotation_path,
    )
    return df


def lookup_annotation(
    annotations: "pd.DataFrame | None",
    id_col: "str | None",
    protein_id: str,
) -> "dict | None":
    """
    Return the annotation fields for a single protein as a plain dict.

    Returns ``None`` when annotations are unavailable or the protein is
    not found (rather than raising, so the benchmark loop stays robust).

    Parameters
    ----------
    annotations : pd.DataFrame | None
        The annotation table returned by :func:`load_annotations`.
    id_col : str | None
        Column to match on.  ``None`` means use the index.
    protein_id : str
        Identifier to look up (must match the strings stored in *id_col*).
    """
    if annotations is None:
        return None

    try:
        if id_col is not None and id_col in annotations.columns:
            matches = annotations[annotations[id_col] == str(protein_id)]
        else:
            # Fallback: try matching on index
            matches = annotations[annotations.index.astype(str) == str(protein_id)]

        if matches.empty:
            return None

        # Return the first matching row as a dict (drop NaN-only columns)
        row = matches.iloc[0].dropna().to_dict()
        return row

    except Exception as exc:  # noqa: BLE001
        logger.debug(
            "lookup_annotation failed for protein_id=%s: %s", protein_id, exc
        )
        return None


# ─── Public entry point ──────────────────────────────────────────────────

def load_dataset(args: argparse.Namespace) -> DatasetBundle:
    """
    Load the full feature matrix, labels, and annotations for the chosen
    dataset and return them in a :class:`DatasetBundle`.

    The function dispatches to the appropriate internal loader based on
    ``args.input_type`` (``'pssm'`` or ``'plm'``), which is populated by
    :func:`resolve_args` from the DATASET_REGISTRY.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed and resolved CLI arguments.  Relevant fields:

        * ``args.dataset``        — name key (e.g. ``'globins'``)
        * ``args.input_type``     — ``'pssm'`` | ``'plm'``
        * ``args.data_path``      — resolved absolute path to data folder
        * ``args.annotation_path``— resolved absolute path (or ``None``)
        * ``args.annotation_id_col`` — column name for protein id (or ``None``)

    Returns
    -------
    DatasetBundle
        A fully validated bundle with ``features``, ``labels``, and
        optionally ``annotations``.

    Raises
    ------
    ValueError
        If ``args.input_type`` is not ``'pssm'`` or ``'plm'``.
    FileNotFoundError
        If the data directory or expected feature files are missing.
    """
    dataset_name = args.dataset
    input_type   = args.input_type
    data_path    = args.data_path

    logger.info(
        "Loading dataset '%s' (input_type=%s) from %s",
        dataset_name, input_type, data_path,
    )

    if input_type == "pssm":
        features, labels = _load_pssm_features(data_path, dataset_name)
    elif input_type == "plm":
        features, labels = _load_plm_features(data_path, dataset_name)
    else:
        raise ValueError(
            f"Unknown input_type '{input_type}' for dataset '{dataset_name}'.\n"
            "Expected 'pssm' or 'plm'.  Check DATASET_REGISTRY."
        )

    annotations = load_annotations(args.annotation_path, args.annotation_id_col)

    return DatasetBundle(
        features=features,
        labels=labels,
        input_type=input_type,
        data_path=data_path,
        annotations=annotations,
        annotation_id_col=args.annotation_id_col,
    )


# ---------------------------------------------------------------------------
# Stub functions (to be implemented in subsequent steps)
# ---------------------------------------------------------------------------



def build_poincare_map(features: np.ndarray, args: argparse.Namespace, tmp_dir: str):
    """
    Build a Poincaré map from a feature matrix.

    Returns
    -------
    embeddings   : np.ndarray, shape (N, 2)
    loss         : float
    build_time   : float  (seconds, wall-clock)
    """
    import torch
    from data import compute_rfa_w_custom_distance
    from model import PoincareEmbedding, PoincareDistance
    from rsgd import RiemannianSGD
    from train import train

    start = time.perf_counter()

    # 1. Compute RFA Target Matrix
    RFA = compute_rfa_w_custom_distance(
        features=features,
        distance_matrix=None,
        k_neighbours=args.knn,
        distfn=args.distfn,
        connected=args.connected,
        sigma=args.sigma,
        distlocal=args.distlocal,
        output_path=tmp_dir,
    )

    # 2. Setup Dataset & Model
    indices = torch.arange(len(RFA))
    dataset = torch.utils.data.TensorDataset(indices, RFA)

    predictor = PoincareEmbedding(
        len(dataset),
        args.dim,
        dist=PoincareDistance,
        gamma=args.gamma,
        lossfn=args.lossfn,
        Qdist=args.distr,
        cuda=False,
    )
    optimizer = RiemannianSGD(predictor.parameters(), lr=args.lr)

    # 3. Setup Optimizer Options for train()
    # The train() function expects an argparse Namespace with specific fields
    class Opt: pass
    opt = Opt()
    N = len(features)
    opt.batchsize = max(4, int(min(128, N // 10)))
    opt.lr = args.lr
    opt.lrm = 1.0
    opt.epochs = args.epochs
    opt.burnin = max(1, args.epochs // 2)
    opt.checkout_freq = 100
    opt.debugplot = 0
    opt.earlystop = args.earlystop
    opt.seed = args.seed

    # 4. Train
    fout_path = os.path.join(tmp_dir, 'tmp_map')
    embeddings, loss, epoch = train(
        predictor, dataset, optimizer, opt, fout=fout_path, earlystop=args.earlystop
    )

    build_time = time.perf_counter() - start
    return np.array(embeddings), float(loss), float(build_time)


def compute_quality(features: np.ndarray, embeddings: np.ndarray, args: argparse.Namespace):
    """
    Compute Qlocal and Qglobal for a given (features, embeddings) pair.

    Returns
    -------
    qlocal  : float
    qglobal : float
    """
    # Reuse the quality implementation from Breton's experimental script
    try:
        from experiments.new_benchmark_once import compute_quality_local
    except ImportError as exc:
        raise ImportError(
            "Could not import 'compute_quality_local' from experiments.new_benchmark_once. "
            "Ensure the project root is in sys.path."
        ) from exc

    qlocal, qglobal, _kmax, _df_score = compute_quality_local(
        coord_high=features,
        coord_low=embeddings,
        setting='manifold',
        k_neighbours=args.k_quality,
        my_metric=args.distlocal,
    )

    return float(qlocal), float(qglobal)


@dataclass
class ReducedMapResult:
    """
    Container for the result of building a Poincaré map on N-1 proteins.
    """
    emb_red: np.ndarray
    feats_reduced: np.ndarray
    labels_reduced: np.ndarray
    removed_feat: np.ndarray
    removed_label: str
    loss_red: float
    build_time: float
    qlocal_red: float
    qglobal_red: float


def build_reduced_map_for_removal(
    remove_idx: int,
    bundle: DatasetBundle,
    args: argparse.Namespace,
    tmp_dir: str,
) -> ReducedMapResult:
    """
    Given the index of a protein to remove, constructs the N-1 feature
    matrix, builds the reduced map, and computes its quality.
    
    Raises exceptions if map building fails, leaving it to the caller
    to catch and log them.
    """
    N = len(bundle)
    mask = np.ones(N, dtype=bool)
    mask[remove_idx] = False
    
    feats_reduced = bundle.features[mask]
    labels_reduced = bundle.labels[mask]
    
    removed_feat = bundle.features[remove_idx]
    removed_label = str(bundle.labels[remove_idx])
    
    # 1. Build the map
    emb_red, loss_red, build_time = build_poincare_map(feats_reduced, args, tmp_dir)
    
    # 2. Compute quality
    qlocal_red, qglobal_red = compute_quality(feats_reduced, emb_red, args)
    
    return ReducedMapResult(
        emb_red=emb_red,
        feats_reduced=feats_reduced,
        labels_reduced=labels_reduced,
        removed_feat=removed_feat,
        removed_label=removed_label,
        loss_red=loss_red,
        build_time=build_time,
        qlocal_red=qlocal_red,
        qglobal_red=qglobal_red,
    )

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
    N = len(embeddings_full)
    if n_remove <= 0 or n_remove >= N:
        logger.info("      n_remove=%d -> selecting ALL %d proteins.", n_remove, N)
        return np.arange(N)

    if mode == "random":
        indices = rng.choice(N, size=n_remove, replace=False)
        logger.info("      Sampled %d proteins randomly.", len(indices))
        return indices
    
    if mode == "stratified_radius":
        radii = np.linalg.norm(embeddings_full, axis=1)
        
        # 3 quantiles: [0, 33%], [33%, 66%], [66%, 100%]
        q33 = np.quantile(radii, 0.3333)
        q66 = np.quantile(radii, 0.6667)
        
        bin1_idx = np.where(radii <= q33)[0]
        bin2_idx = np.where((radii > q33) & (radii <= q66))[0]
        bin3_idx = np.where(radii > q66)[0]
        
        # Determine how many to sample from each bin (approx equally)
        n_per_bin = n_remove // 3
        remainder = n_remove % 3
        
        counts = [n_per_bin] * 3
        for i in range(remainder):
            counts[i] += 1
            
        sampled = []
        for b_idx, count, b_name in zip([bin1_idx, bin2_idx, bin3_idx], counts, ["center", "mid", "periphery"]):
            if count > len(b_idx):
                logger.warning(
                    "      Bin %s has only %d proteins, but %d requested. Sampling all.",
                    b_name, len(b_idx), count
                )
                sample = b_idx
            else:
                sample = rng.choice(b_idx, size=count, replace=False)
            sampled.extend(sample)
            logger.info("      Bin %-9s (n=%3d): sampled %2d proteins", b_name, len(b_idx), len(sample))
            
        indices = np.array(sampled)
        rng.shuffle(indices) # Shuffle the final list so iterations aren't sorted by radius
        return indices

    raise ValueError(f"Unknown sample_mode: {mode}")


def compute_target_vector(
    removed_feat: np.ndarray,
    remaining_feats: np.ndarray,
    gamma: float,
    distlocal: str,
) -> np.ndarray:
    """
    Compute the target RFA-like probability vector for the removed protein
    w.r.t. the remaining proteins' feature space.
    
    This matches the baseline implementation: distances are computed using
    `distlocal` (e.g. cosine) and transformed into weights via a simple
    exponential kernel `exp(-gamma * d)`.

    Returns
    -------
    target : np.ndarray, shape (N-1,)
        Normalized probability distribution summing to 1.
    """
    from sklearn.metrics.pairwise import pairwise_distances
    
    removed_feat_2d = removed_feat.reshape(1, -1)
    
    # 1. Distances to all remaining points in feature space
    d = pairwise_distances(removed_feat_2d, remaining_feats, metric=distlocal).flatten()
    
    # 2. Kernel to probabilities
    target = np.exp(-gamma * d)
    
    # 3. Normalize
    if target.sum() <= 0:
        target = np.ones_like(target) / float(len(target))
    else:
        target = target / target.sum()
        
    return target


def insert_method1_bary(
    model, target: np.ndarray, args: argparse.Namespace
) -> tuple[np.ndarray, float]:
    """
    Method 1: pure hyperbolic barycenter (no gradient optimisation).
    
    Inputs needed:
    - model: An initialized `PoincareEmbedding` populated with the N-1 
             reduced map embeddings (`model.lt.weight`).
    - target: A 1D numpy array of length N-1 representing the target 
              RFA probabilities (weights) of the new point.
    - args: Used to extract `knn`.
              
    (Note: This simply wraps `model.hyperbolic_barycenter`).

    Returns
    -------
    embedding   : np.ndarray, shape (dim,)
    elapsed_sec : float
    """
    import torch
    import time
    
    start = time.perf_counter()
    with torch.no_grad():
        target_tensor = torch.from_numpy(target).float()
        
        # Use top-K neighbors to compute the barycenter, mimicking model.py logic
        k_local = min(max(1, args.knn), target_tensor.numel())
        topk = torch.topk(target_tensor, k=k_local).indices
        
        neighbor_embs = model.lt.weight.data[topk]
        neighbor_w = target_tensor[topk]
        neighbor_w = neighbor_w / neighbor_w.sum()
        
        v = model.hyperbolic_barycenter(
            neighbor_embs, neighbor_w, n_steps=100, tol=1e-7, alpha=1.0, device='cpu'
        )
        
    insertion_time = time.perf_counter() - start
    return v.detach().cpu().numpy().flatten(), insertion_time


def insert_method2_rand(
    model, target: np.ndarray, args: argparse.Namespace
) -> tuple[np.ndarray, float]:
    """
    Method 2: infer_embedding_for_point with random initialisation.
    
    Inputs needed:
    - model: An initialized `PoincareEmbedding` populated with the N-1 
             reduced map embeddings (`model.lt.weight`).
    - target: A 1D numpy array of length N-1 representing the target 
              RFA probabilities.
    - args: Used to extract `n_steps_insert`, `lr_insert`, `k_quality`.
    
    (Note: This simply wraps `model.infer_embedding_for_point(init='random')`).

    Returns
    -------
    embedding   : np.ndarray, shape (dim,)
    elapsed_sec : float
    """
    import torch
    import time
    
    target_tensor = torch.from_numpy(target).float()
    start = time.perf_counter()
    
    new_emb = model.infer_embedding_for_point(
        target_tensor,
        init='random',
        lr=args.lr_insert,
        n_steps=args.n_steps_insert
    )
    
    insertion_time = time.perf_counter() - start
    return new_emb, insertion_time


def insert_method3_bary(
    model, target: np.ndarray, args: argparse.Namespace
) -> tuple[np.ndarray, float]:
    """
    Method 3: infer_embedding_for_point with barycenter initialisation.
    
    Inputs needed:
    - model: An initialized `PoincareEmbedding` populated with the N-1 
             reduced map embeddings (`model.lt.weight`).
    - target: A 1D numpy array of length N-1 representing the target 
              RFA probabilities.
    - args: Used to extract `n_steps_insert`, `lr_insert`, `k_quality`.
    
    (Note: This simply wraps `model.infer_embedding_for_point(init='barycenter')`).

    Returns
    -------
    embedding   : np.ndarray, shape (dim,)
    elapsed_sec : float
    """
    import torch
    import time
    
    target_tensor = torch.from_numpy(target).float()
    start = time.perf_counter()
    
    new_emb = model.infer_embedding_for_point(
        target_tensor,
        init='barycenter',
        lr=args.lr_insert,
        n_steps=args.n_steps_insert
    )
    
    insertion_time = time.perf_counter() - start
    return new_emb, insertion_time


def poincare_pairwise_cross(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute pairwise hyperbolic distances between two sets of points in the
    Poincaré ball.
    x: (N, dim)
    y: (M, dim)
    Returns: (N, M) distance matrix.
    """
    from sklearn.metrics.pairwise import pairwise_distances
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    norm2_x = np.sum(x * x, axis=1)
    norm2_y = np.sum(y * y, axis=1)
    sqd = pairwise_distances(x, y, metric='euclidean') ** 2
    denom = (1.0 - norm2_x)[:, None] * (1.0 - norm2_y)[None, :]
    denom = np.maximum(denom, 1e-12)
    arg = 1.0 + 2.0 * sqd / denom
    arg = np.maximum(arg, 1.0)
    d = np.log(arg + np.sqrt((arg - 1.0) * (arg + 1.0)))
    return d


def compute_neighbor_overlap(
    full_emb: np.ndarray,
    removed_idx: int,
    inserted_emb: np.ndarray,
    reduced_emb: np.ndarray,
    k: int,
) -> float:
    """
    Compare the k nearest neighbours of the removed protein in the full map
    vs. the k nearest neighbours of the inserted point in the reduced map.

    Returns the fraction of shared neighbours (Jaccard-style overlap ∈ [0, 1]).
    """
    # 1. Distances from the original point to all OTHER points in full map
    emb_full_others = np.delete(full_emb, removed_idx, axis=0)
    x_orig = full_emb[removed_idx].reshape(1, -1)
    d_orig = poincare_pairwise_cross(x_orig, emb_full_others).flatten()
    
    # Indices of top-k neighbors in emb_full_others (which structurally align with reduced_emb)
    top_k_orig = np.argsort(d_orig)[:k]
    
    # 2. Distances from inserted point to reduced_emb
    x_ins = inserted_emb.reshape(1, -1)
    d_ins = poincare_pairwise_cross(x_ins, reduced_emb).flatten()
    
    top_k_ins = np.argsort(d_ins)[:k]
    
    # 3. Overlap
    intersection = len(np.intersect1d(top_k_orig, top_k_ins))
    return float(intersection / k)




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
    logger.info("[1/5] Loading features ...")
    bundle = load_dataset(args)
    features = bundle.features
    labels   = bundle.labels
    N = len(bundle)
    logger.info(
        "      Loaded %d proteins, feature dim=%d  (input_type=%s)",
        N, bundle.feature_dim(), bundle.input_type,
    )

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

    # Annotations were loaded inside load_dataset and stored in the bundle.
    if bundle.annotations is not None:
        logger.info("      Annotations available (%d rows)", len(bundle.annotations))

    # ------------------------------------------------------------------
    # Step 4: Leave-one-out loop
    # ------------------------------------------------------------------
    logger.info("[4/5] Starting leave-one-out iterations …")
    all_rows: list[dict] = []
    n_ok = 0
    n_err = 0

    for iter_idx, remove_idx in enumerate(remove_indices):
        protein_id = str(bundle.labels[remove_idx])
        logger.info(
            "  [iter %d/%d]  protein=%s  radius=%.4f",
            iter_idx + 1, len(remove_indices), protein_id, radii_full[remove_idx],
        )

        # --- a) Build reduced map (N-1 proteins) ----------------------
        try:
            tmp_red = os.path.join(paths["base"], f"_tmp_reduced_{iter_idx:04d}")
            os.makedirs(tmp_red, exist_ok=True)

            reduced_res = build_reduced_map_for_removal(remove_idx, bundle, args, tmp_red)

        except Exception as exc:
            logger.warning("    Reduced map FAILED for %s: %s", protein_id, exc)
            log_error(paths["errors"], args.dataset, protein_id, "build_reduced_map", exc)
            n_err += 1
            continue

        # --- b) Compute insertion target vector -----------------------
        target = compute_target_vector(
            reduced_res.removed_feat, reduced_res.feats_reduced, args.gamma, args.distlocal
        )

        # --- c) Instantiate model loaded with reduced embeddings ------
        import torch
        from model import PoincareEmbedding
        
        dim = reduced_res.emb_red.shape[1]
        model = PoincareEmbedding(
            size=reduced_res.emb_red.shape[0],
            dim=dim,
            gamma=args.gamma,
            lossfn=args.lossfn,
            Qdist=args.distr,
            cuda=False
        )
        # Load the reduced map geometry into the model
        with torch.no_grad():
            model.lt.weight.data = torch.from_numpy(reduced_res.emb_red).float()

        # --- d-f) Run each insertion method ---------------------------
        methods = {
            "bary_init":  lambda: insert_method1_bary(model, target, args),
            "infer_rand": lambda: insert_method2_rand(model, target, args),
            "infer_bary": lambda: insert_method3_bary(model, target, args),
        }

        for method_name, method_fn in methods.items():
            try:
                new_emb, insertion_time = method_fn()

                # --- g) Q-metrics after insertion ---------------------
                # Reconstruct combined embedding: N-1 reduced + 1 inserted
                # (position of the removed protein inserted at remove_idx)
                emb_after = np.insert(reduced_res.emb_red, remove_idx, new_emb.reshape(1, -1), axis=0)
                qlocal_after, qglobal_after = compute_quality(features, emb_after, args)

                # --- h) Neighbour overlap & density proxy -------------
                inserted_radius = float(np.linalg.norm(new_emb))
                neighbor_overlaps: dict[int, float] = {}
                for k in args.neighbor_overlap_k:
                    neighbor_overlaps[k] = compute_neighbor_overlap(
                        emb_full, remove_idx, new_emb, reduced_res.emb_red, k
                    )

                # Local density proxy: mean hyperbolic distance to top-5
                # neighbours in the full map.
                x_orig = emb_full[remove_idx].reshape(1, -1)
                emb_full_others = np.delete(emb_full, remove_idx, axis=0)
                d_orig = poincare_pairwise_cross(x_orig, emb_full_others).flatten()
                local_density_proxy = float(np.mean(np.sort(d_orig)[:5]))

                # Annotation fields
                annotation_row: dict | None = None
                if bundle.annotations is not None:
                    # look up by protein_id — deferred to implementation step
                    pass

                row = build_result_row(
                    dataset=args.dataset,
                    protein_id=protein_id,
                    method=method_name,
                    sample_mode=args.sample_mode,
                    map_build_time=reduced_res.build_time,
                    insertion_time=insertion_time,
                    qlocal_reduced_before=reduced_res.qlocal_red,
                    qglobal_reduced_before=reduced_res.qglobal_red,
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
                    qlocal_after - reduced_res.qlocal_red,
                    qglobal_after - reduced_res.qglobal_red,
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
