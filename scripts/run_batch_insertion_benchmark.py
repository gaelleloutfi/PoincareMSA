#!/usr/bin/env python3
"""
Batch Insertion Benchmark for Poincaré Map
==========================================
Compares four strategies for inserting k proteins simultaneously into a
Poincaré disk map trained on the remaining N-k proteins.

Strategies
----------
  independent      — each protein uses only N-k map anchors; no coupling
  seq_center_first — sequential barycenter, low-radius proteins inserted first
  seq_peri_first   — sequential barycenter, high-radius proteins inserted first
  iterative        — iterate barycenter over all batch members until convergence

For every batch the key metric is:

    delta_vs_full_qlocal  = Qlocal_after_N_pts − Qlocal_full_reference

A value of 0 means perfect recovery of the reference map quality.

Usage
-----
  python scripts/run_batch_insertion_benchmark.py \\
      --dataset globins \\
      --batch_sizes 5 10 20 \\
      --n_batches 5 \\
      --output_dir benchmark_results \\
      --seed 42
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_SCRIPT_DIR  = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_BUILD_PKG   = _PROJECT_ROOT / "scripts" / "build_poincare_map"

for _p in [str(_PROJECT_ROOT), str(_BUILD_PKG), str(_SCRIPT_DIR)]:
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
logger = logging.getLogger("batch_benchmark")

# ---------------------------------------------------------------------------
# Reuse shared infrastructure from the LOO benchmark
# ---------------------------------------------------------------------------
import run_leave_one_out_benchmark as loo

DATASET_REGISTRY      = loo.DATASET_REGISTRY
DatasetBundle         = loo.DatasetBundle
load_dataset          = loo.load_dataset
build_poincare_map    = loo.build_poincare_map
compute_quality       = loo.compute_quality
compute_target_vector = loo.compute_target_vector
resolve_args          = loo.resolve_args


# ---------------------------------------------------------------------------
# Batch map result container
# ---------------------------------------------------------------------------

@dataclass
class BatchMapResult:
    """Poincaré map built from the N-k proteins remaining after the batch is removed."""
    emb_red:      np.ndarray   # (N-k, 2)
    feats_reduced: np.ndarray  # (N-k, D)
    mask:         np.ndarray   # (N,) bool  — True = in the N-k map
    loss_red:     float
    build_time:   float
    qlocal_red:   float
    qglobal_red:  float


# ---------------------------------------------------------------------------
# Build N-k reduced map
# ---------------------------------------------------------------------------

def build_reduced_map_for_batch(
    batch_indices: np.ndarray,
    bundle: DatasetBundle,
    args: argparse.Namespace,
    tmp_dir: str,
) -> BatchMapResult:
    N    = len(bundle)
    mask = np.ones(N, dtype=bool)
    mask[batch_indices] = False

    feats_reduced = bundle.features[mask]
    emb_red, loss_red, build_time = build_poincare_map(feats_reduced, args, tmp_dir)
    qlocal_red, qglobal_red       = compute_quality(feats_reduced, emb_red, args)

    return BatchMapResult(
        emb_red=emb_red,
        feats_reduced=feats_reduced,
        mask=mask,
        loss_red=loss_red,
        build_time=build_time,
        qlocal_red=qlocal_red,
        qglobal_red=qglobal_red,
    )


# ---------------------------------------------------------------------------
# Core insertion primitive
# ---------------------------------------------------------------------------

def _bary_insert(
    feat: np.ndarray,
    feats_pool: np.ndarray,
    embs_pool: np.ndarray,
    model,
    knn: int,
    gamma: float,
    distlocal: str,
) -> np.ndarray:
    """
    Hyperbolic-barycenter position for one protein.

    feat       : (D,)   feature vector of the protein to insert
    feats_pool : (M, D) features of all available anchors
    embs_pool  : (M, 2) embeddings of those anchors
    """
    target        = compute_target_vector(feat, feats_pool, gamma, distlocal)
    target_tensor = torch.from_numpy(target).float()

    k_local  = min(max(1, knn), len(target))
    topk_idx = torch.topk(target_tensor, k=k_local).indices.numpy()

    neighbor_embs = torch.from_numpy(embs_pool[topk_idx]).float()
    neighbor_w    = target_tensor[topk_idx]
    neighbor_w    = neighbor_w / neighbor_w.sum()

    with torch.no_grad():
        v = model.hyperbolic_barycenter(
            neighbor_embs, neighbor_w, n_steps=100, tol=1e-7, alpha=1.0, device="cpu"
        )
    return v.detach().cpu().numpy().flatten()


# ---------------------------------------------------------------------------
# Batch insertion strategies
# ---------------------------------------------------------------------------

def insert_batch_independent(
    batch_feats: np.ndarray,
    batch_result: BatchMapResult,
    model,
    args: argparse.Namespace,
) -> tuple[np.ndarray, float]:
    """
    Strategy: independent.
    Each protein is placed using only the N-k base map as anchors.
    Batch members are completely unaware of each other.
    """
    positions = []
    t0 = time.perf_counter()
    for feat in batch_feats:
        pos = _bary_insert(
            feat,
            batch_result.feats_reduced,
            batch_result.emb_red,
            model, args.knn, args.gamma, args.distlocal,
        )
        positions.append(pos)
    return np.array(positions), time.perf_counter() - t0


def insert_batch_sequential(
    batch_feats: np.ndarray,
    batch_result: BatchMapResult,
    model,
    args: argparse.Namespace,
    order: np.ndarray,
) -> tuple[np.ndarray, float]:
    """
    Strategy: sequential.
    Proteins are placed one at a time in *order* (indices into batch_feats).
    Each protein sees the already-inserted batch members as additional anchors
    in both feature space and embedding space.
    """
    k          = len(batch_feats)
    positions  = [None] * k
    t0         = time.perf_counter()

    for step, batch_idx in enumerate(order):
        feat = batch_feats[batch_idx]

        if step == 0:
            feats_pool = batch_result.feats_reduced
            embs_pool  = batch_result.emb_red
        else:
            prev_idxs  = order[:step]
            extra_feats = np.vstack([batch_feats[i].reshape(1, -1) for i in prev_idxs])
            extra_embs  = np.vstack([positions[i].reshape(1, -1)   for i in prev_idxs])
            feats_pool  = np.vstack([batch_result.feats_reduced, extra_feats])
            embs_pool   = np.vstack([batch_result.emb_red,        extra_embs])

        positions[batch_idx] = _bary_insert(
            feat, feats_pool, embs_pool, model, args.knn, args.gamma, args.distlocal,
        )

    return np.array(positions), time.perf_counter() - t0


def insert_batch_iterative(
    batch_feats: np.ndarray,
    batch_result: BatchMapResult,
    model,
    args: argparse.Namespace,
    max_iters: int = 15,
    tol: float = 1e-5,
) -> tuple[np.ndarray, float, int]:
    """
    Strategy: iterative refinement.
    1. Initialise with independent positions.
    2. Each round: for every protein i, recompute its barycenter using the
       N-k base map plus all *other* batch proteins at their current positions.
    3. Stop when the maximum positional shift < tol or max_iters is reached.

    Returns (positions, total_time, n_rounds_performed).
    """
    t0 = time.perf_counter()

    positions, _ = insert_batch_independent(batch_feats, batch_result, model, args)

    n_iters = 0
    for it in range(max_iters):
        new_positions = np.zeros_like(positions)
        for i, feat in enumerate(batch_feats):
            other_mask  = np.ones(len(batch_feats), dtype=bool)
            other_mask[i] = False
            feats_pool  = np.vstack([batch_result.feats_reduced, batch_feats[other_mask]])
            embs_pool   = np.vstack([batch_result.emb_red,        positions[other_mask]])

            new_positions[i] = _bary_insert(
                feat, feats_pool, embs_pool, model, args.knn, args.gamma, args.distlocal,
            )

        max_shift = float(np.max(np.linalg.norm(new_positions - positions, axis=1)))
        positions = new_positions
        n_iters   = it + 1
        if max_shift < tol:
            logger.info("        Iterative converged after %d rounds (shift=%.2e)", n_iters, max_shift)
            break
    else:
        logger.info("        Iterative: %d rounds, not fully converged", max_iters)

    return positions, time.perf_counter() - t0, n_iters


def insert_batch_joint_sgd(
    batch_feats: np.ndarray,
    batch_result: BatchMapResult,
    model,
    args: argparse.Namespace,
    n_steps: int = 500,
    lr: float = 0.05,
) -> tuple[np.ndarray, float, int]:
    """
    Strategy: joint SGD.
    All k new positions are optimized simultaneously as shared parameters.
    Each protein i's loss is computed against the N-k base anchors PLUS the
    other k-1 batch proteins at their current (moving) positions, so gradients
    couple all k points in every step.

    Initialised from independent barycenter positions (warm start).
    Returns (positions, total_time, n_steps_run).
    """
    import torch

    t0 = time.perf_counter()
    k  = len(batch_feats)

    # Pre-compute feature-space targets (fixed throughout the SGD)
    # target[i] has length N-k + k-1 = N-1, covering:
    #   [base proteins 0..N-k-1, batch protein j for j != i in order]
    targets = []
    for i, feat in enumerate(batch_feats):
        other_mask  = np.ones(k, dtype=bool); other_mask[i] = False
        other_feats = batch_feats[other_mask]
        pool_feats  = np.vstack([batch_result.feats_reduced, other_feats])
        t = compute_target_vector(feat, pool_feats, args.gamma, args.distlocal)
        targets.append(t)

    # Warm start: independent barycenter positions
    init_positions, _ = insert_batch_independent(batch_feats, batch_result, model, args)

    # Temporarily set the model embedding table to the N-k base embeddings
    # (infer_batch_embedding reads self.lt.weight as the fixed anchor pool)
    model.lt.weight.data = torch.from_numpy(batch_result.emb_red).float()

    positions = model.infer_batch_embedding(
        targets=targets,
        n_steps=n_steps,
        lr=lr,
        init_vecs=init_positions,
    )

    return positions, time.perf_counter() - t0, n_steps


# ---------------------------------------------------------------------------
# Full N-point reconstruction
# ---------------------------------------------------------------------------

def reconstruct_full_embedding(
    batch_indices: np.ndarray,
    inserted_embs: np.ndarray,
    batch_result:  BatchMapResult,
    N: int,
) -> np.ndarray:
    """Reconstruct the N-point embedding matrix from the N-k base + k inserted positions."""
    dim        = batch_result.emb_red.shape[1]
    full_embs  = np.zeros((N, dim))
    full_embs[batch_result.mask] = batch_result.emb_red
    for j, orig_idx in enumerate(batch_indices):
        full_embs[orig_idx] = inserted_embs[j]
    return full_embs


# ---------------------------------------------------------------------------
# Per-protein neighborhood overlap
# ---------------------------------------------------------------------------

def neighbor_overlap(full_embs_ref: np.ndarray, full_embs_after: np.ndarray, idx: int, k: int) -> float:
    """Fraction of top-k neighbors of *idx* shared between reference and inserted map."""
    N = len(full_embs_ref)
    if N < k + 1:
        return float("nan")

    def topk(embs, i):
        d = np.linalg.norm(embs - embs[i], axis=1)
        d[i] = np.inf
        return set(np.argsort(d)[:k])

    return len(topk(full_embs_ref, idx) & topk(full_embs_after, idx)) / k


# ---------------------------------------------------------------------------
# Result row builder
# ---------------------------------------------------------------------------

def build_batch_rows(
    dataset:             str,
    batch_id:            int,
    batch_size:          int,
    strategy:            str,
    ordering:            str,
    n_iters:             int,
    batch_indices:       np.ndarray,
    bundle:              DatasetBundle,
    batch_result:        BatchMapResult,
    full_embs_after:     np.ndarray,
    full_embs_ref:       np.ndarray,
    qlocal_full_ref:     float,
    qglobal_full_ref:    float,
    qlocal_after:        float,
    qglobal_after:       float,
    total_insertion_time: float,
) -> list[dict]:
    """One row per inserted protein; batch-level metrics are repeated on every row."""
    full_ref_radii = np.linalg.norm(full_embs_ref, axis=1)
    rows = []
    for j, orig_idx in enumerate(batch_indices):
        rows.append({
            "dataset":                  dataset,
            "batch_id":                 batch_id,
            "batch_size":               batch_size,
            "strategy":                 strategy,
            "ordering":                 ordering,
            "n_iters":                  n_iters,
            "protein_id":               str(bundle.labels[orig_idx]),
            # Batch-level quality (same for all proteins in this batch × strategy)
            "map_build_time":           batch_result.build_time,
            "total_insertion_time":     total_insertion_time,
            "qlocal_red":               batch_result.qlocal_red,
            "qglobal_red":              batch_result.qglobal_red,
            "qlocal_full_ref":          qlocal_full_ref,
            "qglobal_full_ref":         qglobal_full_ref,
            "qlocal_after":             qlocal_after,
            "qglobal_after":            qglobal_after,
            "delta_vs_reduced_qlocal":  qlocal_after  - batch_result.qlocal_red,
            "delta_vs_reduced_qglobal": qglobal_after - batch_result.qglobal_red,
            "delta_vs_full_qlocal":     qlocal_after  - qlocal_full_ref,
            "delta_vs_full_qglobal":    qglobal_after - qglobal_full_ref,
            # Per-protein placement metrics
            "full_map_radius":          float(full_ref_radii[orig_idx]),
            "inserted_radius":          float(np.linalg.norm(full_embs_after[orig_idx])),
            "neighbor_overlap_k5":      neighbor_overlap(full_embs_ref, full_embs_after, orig_idx, 5),
            "neighbor_overlap_k10":     neighbor_overlap(full_embs_ref, full_embs_after, orig_idx, 10),
        })
    return rows


# ---------------------------------------------------------------------------
# Batch sampling
# ---------------------------------------------------------------------------

def sample_batch(
    full_embs:  np.ndarray,
    batch_size: int,
    rng:        np.random.Generator,
) -> np.ndarray:
    """Stratified sample of batch_size proteins across radial bins (center/mid/periphery)."""
    radii = np.linalg.norm(full_embs, axis=1)
    q33   = np.quantile(radii, 0.3333)
    q66   = np.quantile(radii, 0.6667)

    bins   = [
        np.where(radii <= q33)[0],
        np.where((radii > q33) & (radii <= q66))[0],
        np.where(radii > q66)[0],
    ]
    n_base = batch_size // 3
    counts = [n_base, n_base, n_base]
    for i in range(batch_size % 3):
        counts[i] += 1

    sampled = []
    for bin_idx, count in zip(bins, counts):
        count = min(count, len(bin_idx))
        sampled.extend(rng.choice(bin_idx, size=count, replace=False))
    return np.array(sampled)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Batch insertion benchmark for Poincaré maps.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset", required=True, choices=list(DATASET_REGISTRY))
    p.add_argument("--data_path",        default=None)
    p.add_argument("--annotation_path",  default=None)
    p.add_argument("--batch_sizes",      type=int, nargs="+", default=[5, 10, 20])
    p.add_argument("--n_batches",        type=int, default=5,   help="Batches per batch size.")
    p.add_argument("--seed",             type=int, default=42)
    p.add_argument("--output_dir",       default="benchmark_results")
    p.add_argument("--iterative_max_iters", type=int, default=15)
    p.add_argument("--joint_sgd_steps",    type=int,   default=500,  help="SGD steps for joint_sgd strategy.")
    p.add_argument("--joint_sgd_lr",       type=float, default=0.05, help="Learning rate for joint_sgd strategy.")
    p.add_argument(
        "--only_strategies", type=str, nargs="+", default=None,
        help="Run only these strategies (e.g. --only_strategies joint_sgd). Default: run all.",
    )

    # Map hyperparameters (same defaults as LOO benchmark)
    p.add_argument("--epochs",    type=int,   default=300)
    p.add_argument("--knn",       type=int,   default=None)
    p.add_argument("--sigma",     type=float, default=None)
    p.add_argument("--gamma",     type=float, default=None)
    p.add_argument("--lr",        type=float, default=0.1)
    p.add_argument("--dim",       type=int,   default=2)
    p.add_argument("--earlystop", type=float, default=0.0)
    p.add_argument("--connected", type=int,   default=1, choices=[0, 1])
    p.add_argument("--k_quality", type=int,   default=5)
    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = build_parser().parse_args()
    args = resolve_args(args)

    rng      = np.random.default_rng(args.seed)
    out_dir  = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results_path = out_dir / "batch_insertion_results.csv"
    partial_path = out_dir / "batch_insertion_results_partial.csv"
    partial_written = partial_path.exists()   # track whether header was already written

    # ------------------------------------------------------------------
    # 1. Load dataset
    # ------------------------------------------------------------------
    logger.info("[1/4] Loading dataset: %s", args.dataset)
    bundle = load_dataset(args)
    logger.info("      N=%d proteins, D=%d features", len(bundle), bundle.feature_dim())

    # ------------------------------------------------------------------
    # 2. Build full reference map
    # ------------------------------------------------------------------
    logger.info("[2/4] Building full reference map (%d proteins)…", len(bundle))
    tmp_full = str(out_dir / "_tmp_batch_full")
    os.makedirs(tmp_full, exist_ok=True)

    full_embs, full_loss, full_time = build_poincare_map(bundle.features, args, tmp_full)
    qlocal_full_ref, qglobal_full_ref = compute_quality(bundle.features, full_embs, args)
    logger.info(
        "      loss=%.4f  Qlocal=%.4f  Qglobal=%.4f  t=%.1fs",
        full_loss, qlocal_full_ref, qglobal_full_ref, full_time,
    )

    # ------------------------------------------------------------------
    # 3. Load model class (used only for hyperbolic_barycenter)
    # ------------------------------------------------------------------
    from model import PoincareEmbedding, PoincareDistance

    # ------------------------------------------------------------------
    # 4. Batch loop
    # ------------------------------------------------------------------
    logger.info("[3/4] Running batch insertion experiments…")
    all_rows: list[dict] = []

    total_exps = len(args.batch_sizes) * args.n_batches
    exp_idx    = 0

    for batch_size in args.batch_sizes:
        logger.info("  === Batch size k=%d ===", batch_size)
        if batch_size >= len(bundle):
            logger.warning("      batch_size=%d >= N=%d, skipping.", batch_size, len(bundle))
            continue

        for batch_id in range(args.n_batches):
            exp_idx += 1
            logger.info("  [%d/%d] batch_id=%d  k=%d", exp_idx, total_exps, batch_id, batch_size)

            # Sample batch proteins (stratified by radius in full reference map)
            batch_indices = sample_batch(full_embs, batch_size, rng)
            batch_feats   = bundle.features[batch_indices]
            logger.info(
                "      Proteins: %s",
                ", ".join(str(bundle.labels[i]) for i in batch_indices),
            )

            # Build N-k reduced map
            tmp_red = str(out_dir / f"_tmp_batch_{batch_size}_{batch_id:03d}")
            os.makedirs(tmp_red, exist_ok=True)
            try:
                batch_result = build_reduced_map_for_batch(batch_indices, bundle, args, tmp_red)
            except Exception as exc:
                logger.error("      Reduced map FAILED: %s — skipping batch.", exc)
                continue

            logger.info(
                "      N-k map: Qlocal=%.4f  Qglobal=%.4f  t=%.1fs",
                batch_result.qlocal_red, batch_result.qglobal_red, batch_result.build_time,
            )

            # Instantiate model (embedding table = N-k positions)
            model = PoincareEmbedding(
                len(batch_result.emb_red), args.dim,
                dist=PoincareDistance, gamma=args.gamma,
                lossfn="klSym", Qdist="laplace", cuda=False,
            )
            model.lt.weight.data = torch.from_numpy(batch_result.emb_red).float()

            # Ordering for sequential strategies (by radius in full reference map)
            batch_radii        = np.linalg.norm(full_embs[batch_indices], axis=1)
            order_center_first = np.argsort(batch_radii)
            order_peri_first   = np.argsort(batch_radii)[::-1].copy()

            # All strategies: (name, callable → (positions, time[, n_iters]), ordering_label)
            def run_independent():
                p, t = insert_batch_independent(batch_feats, batch_result, model, args)
                return p, t, 0

            def run_seq_center():
                p, t = insert_batch_sequential(batch_feats, batch_result, model, args, order_center_first)
                return p, t, 0

            def run_seq_peri():
                p, t = insert_batch_sequential(batch_feats, batch_result, model, args, order_peri_first)
                return p, t, 0

            def run_iterative():
                return insert_batch_iterative(batch_feats, batch_result, model, args, args.iterative_max_iters)

            def run_joint_sgd():
                return insert_batch_joint_sgd(
                    batch_feats, batch_result, model, args,
                    n_steps=args.joint_sgd_steps, lr=args.joint_sgd_lr,
                )

            all_strategies = [
                ("independent",      run_independent, "none"),
                ("seq_center_first", run_seq_center,  "center_first"),
                ("seq_peri_first",   run_seq_peri,    "peri_first"),
                ("iterative",        run_iterative,   "none"),
                ("joint_sgd",        run_joint_sgd,   "none"),
            ]
            if args.only_strategies:
                strategies = [(n, fn, o) for n, fn, o in all_strategies if n in args.only_strategies]
            else:
                strategies = all_strategies

            for strategy_name, strategy_fn, ordering in strategies:
                logger.info("      Strategy: %s", strategy_name)
                try:
                    inserted_embs, ins_time, n_iters = strategy_fn()

                    # Reconstruct the full N-point embedding
                    full_embs_after = reconstruct_full_embedding(
                        batch_indices, inserted_embs, batch_result, len(bundle)
                    )

                    # Compute quality of the full N-point map
                    qlocal_after, qglobal_after = compute_quality(bundle.features, full_embs_after, args)

                    rows = build_batch_rows(
                        dataset=args.dataset,
                        batch_id=batch_id,
                        batch_size=batch_size,
                        strategy=strategy_name,
                        ordering=ordering,
                        n_iters=n_iters,
                        batch_indices=batch_indices,
                        bundle=bundle,
                        batch_result=batch_result,
                        full_embs_after=full_embs_after,
                        full_embs_ref=full_embs,
                        qlocal_full_ref=qlocal_full_ref,
                        qglobal_full_ref=qglobal_full_ref,
                        qlocal_after=qlocal_after,
                        qglobal_after=qglobal_after,
                        total_insertion_time=ins_time,
                    )
                    all_rows.extend(rows)

                    # Append to partial checkpoint
                    pd.DataFrame(rows).to_csv(
                        partial_path, mode="a", index=False, header=not partial_written
                    )
                    partial_written = True

                    logger.info(
                        "        Qlocal_after=%.4f  dVsReduced=%+.4f  dVsFull=%+.4f  t=%.2fs  iters=%d",
                        qlocal_after,
                        qlocal_after - batch_result.qlocal_red,
                        qlocal_after - qlocal_full_ref,
                        ins_time,
                        n_iters,
                    )

                except Exception as exc:
                    logger.error("        FAILED (%s): %s", strategy_name, exc)
                    import traceback; traceback.print_exc()

    # ------------------------------------------------------------------
    # 5. Save final results
    # ------------------------------------------------------------------
    logger.info("[4/4] Saving results…")
    if not all_rows:
        logger.warning("No results to save.")
        return

    df = pd.DataFrame(all_rows)

    # Merge with existing results, keeping rows for other datasets AND other
    # strategies within the same dataset (so --only_strategies runs are additive).
    strategies_run = {s for s, _, _ in (
        [(n, fn, o) for n, fn, o in [
            ("independent", None, None), ("seq_center_first", None, None),
            ("seq_peri_first", None, None), ("iterative", None, None),
            ("joint_sgd", None, None),
        ] if args.only_strategies is None or n in args.only_strategies]
    )}
    if results_path.exists():
        try:
            prev = pd.read_csv(results_path)
            keep = prev[~(
                (prev["dataset"] == args.dataset) &
                (prev["strategy"].isin(strategies_run))
            )]
            if len(keep):
                df = pd.concat([keep, df], ignore_index=True)
                logger.info("      Merged %d existing rows.", len(keep))
        except Exception as exc:
            logger.warning("      Could not merge existing results, overwriting: %s", exc)

    df.to_csv(results_path, index=False)
    logger.info("      Saved %d rows → %s", len(df), results_path)
    _save_summary(df[df["dataset"] == args.dataset], out_dir)


def _save_summary(df: pd.DataFrame, out_dir: Path) -> None:
    """Aggregate quality by (strategy, batch_size) — one row per unique combination."""
    # Since each batch contributes batch_size identical rows for batch-level metrics,
    # we first deduplicate to (batch_id, strategy, batch_size) for quality metrics,
    # then average across batches.
    batch_level = df.drop_duplicates(subset=["dataset", "batch_id", "batch_size", "strategy"])

    agg = (
        batch_level.groupby(["dataset", "strategy", "batch_size"])
        .agg(
            n_batches                   = ("batch_id", "nunique"),
            qlocal_full_ref             = ("qlocal_full_ref",           "mean"),
            qlocal_red_mean             = ("qlocal_red",                "mean"),
            qlocal_after_mean           = ("qlocal_after",              "mean"),
            delta_vs_full_qlocal_mean   = ("delta_vs_full_qlocal",      "mean"),
            delta_vs_full_qlocal_std    = ("delta_vs_full_qlocal",      "std"),
            delta_vs_reduced_qlocal_mean = ("delta_vs_reduced_qlocal",  "mean"),
            delta_vs_full_qglobal_mean  = ("delta_vs_full_qglobal",     "mean"),
            insertion_time_mean         = ("total_insertion_time",       "mean"),
            n_iters_mean                = ("n_iters",                    "mean"),
        )
        .reset_index()
    )

    # Per-protein neighbor overlap: averaged over all proteins × batches
    per_prot = df.groupby(["dataset", "strategy", "batch_size"]).agg(
        neighbor_overlap_k5_mean  = ("neighbor_overlap_k5",  "mean"),
        neighbor_overlap_k10_mean = ("neighbor_overlap_k10", "mean"),
    ).reset_index()

    summary = agg.merge(per_prot, on=["dataset", "strategy", "batch_size"])

    reports_dir = out_dir / "reports"
    reports_dir.mkdir(exist_ok=True)
    summary_path = reports_dir / "batch_insertion_summary.csv"
    summary.to_csv(summary_path, index=False)
    logger.info("      Summary → %s", summary_path)


if __name__ == "__main__":
    main()
