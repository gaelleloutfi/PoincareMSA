"""
Core protein insertion functions for Poincaré disk maps.

This module is the production counterpart of run_batch_insertion_benchmark.py.
It exposes a single high-level entry point::

    positions, elapsed = insert_proteins(
        new_feats, existing_feats, existing_embs,
        gamma=2.0, knn=5, distlocal="cosine",
        strategy="independent",
    )

Supported strategies
--------------------
independent      — fastest; each protein inserted independently from base map
seq_center_first — sequential barycenter, small-radius proteins first
seq_peri_first   — sequential barycenter, large-radius proteins first
iterative        — fixed-point refinement using all batch members as mutual anchors

Benchmark finding: all four strategies produce essentially identical quality.
"independent" is recommended (fastest, no coupling overhead).
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.metrics.pairwise import pairwise_distances

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

logger = logging.getLogger(__name__)

STRATEGIES = ("independent", "seq_center_first", "seq_peri_first", "iterative")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_model(dim: int = 2, gamma: float = 2.0):
    """Return a minimal PoincareEmbedding used only for geometric operations."""
    from model import PoincareEmbedding, PoincareDistance
    return PoincareEmbedding(size=1, dim=dim, dist=PoincareDistance, gamma=gamma, lossfn="klSym")


def compute_target_vector(
    new_feat: np.ndarray,
    existing_feats: np.ndarray,
    gamma: float,
    distlocal: str = "cosine",
) -> np.ndarray:
    """
    Compute the RFA-like probability vector for a new protein.

    Parameters
    ----------
    new_feat : (D,)
    existing_feats : (N, D)
    gamma : kernel bandwidth (must match map training value)
    distlocal : pairwise distance metric ('cosine', 'euclidean', ...)

    Returns
    -------
    target : (N,) normalized probability distribution summing to 1
    """
    feat_2d = new_feat.reshape(1, -1)
    d = pairwise_distances(feat_2d, existing_feats, metric=distlocal).flatten()
    target = np.exp(-gamma * d)
    if target.sum() <= 0:
        target = np.ones_like(target) / len(target)
    else:
        target /= target.sum()
    return target


def _bary_insert(
    feat: np.ndarray,
    feats_pool: np.ndarray,
    embs_pool: np.ndarray,
    model,
    knn: int,
    gamma: float,
    distlocal: str,
) -> np.ndarray:
    """Compute the hyperbolic barycenter position for one protein. Returns (2,)."""
    target = compute_target_vector(feat, feats_pool, gamma, distlocal)
    target_t = torch.from_numpy(target).float()

    k_local = min(max(1, knn), len(target))
    topk_idx = torch.topk(target_t, k=k_local).indices.numpy()

    neighbor_embs = torch.from_numpy(embs_pool[topk_idx]).float()
    neighbor_w = target_t[topk_idx]
    neighbor_w = neighbor_w / neighbor_w.sum()

    with torch.no_grad():
        v = model.hyperbolic_barycenter(
            neighbor_embs, neighbor_w, n_steps=100, tol=1e-7, alpha=1.0, device="cpu"
        )
    return v.detach().cpu().numpy().flatten()


# ---------------------------------------------------------------------------
# Strategy implementations (private)
# ---------------------------------------------------------------------------

def _insert_independent(
    new_feats: np.ndarray,
    existing_feats: np.ndarray,
    existing_embs: np.ndarray,
    model,
    knn: int,
    gamma: float,
    distlocal: str,
) -> np.ndarray:
    positions = []
    for feat in new_feats:
        pos = _bary_insert(feat, existing_feats, existing_embs, model, knn, gamma, distlocal)
        positions.append(pos)
    return np.array(positions)


def _insert_sequential(
    new_feats: np.ndarray,
    existing_feats: np.ndarray,
    existing_embs: np.ndarray,
    model,
    knn: int,
    gamma: float,
    distlocal: str,
    order: np.ndarray,
) -> np.ndarray:
    k = len(new_feats)
    positions: list[np.ndarray | None] = [None] * k

    for step, idx in enumerate(order):
        feat = new_feats[idx]
        if step == 0:
            feats_pool = existing_feats
            embs_pool = existing_embs
        else:
            prev_idxs = order[:step]
            extra_feats = np.vstack([new_feats[i].reshape(1, -1) for i in prev_idxs])
            extra_embs = np.vstack([positions[i].reshape(1, -1) for i in prev_idxs])  # type: ignore[arg-type]
            feats_pool = np.vstack([existing_feats, extra_feats])
            embs_pool = np.vstack([existing_embs, extra_embs])

        positions[idx] = _bary_insert(feat, feats_pool, embs_pool, model, knn, gamma, distlocal)

    return np.array(positions)


def _insert_iterative(
    new_feats: np.ndarray,
    existing_feats: np.ndarray,
    existing_embs: np.ndarray,
    model,
    knn: int,
    gamma: float,
    distlocal: str,
    max_iters: int = 15,
    tol: float = 1e-5,
) -> np.ndarray:
    # Warm-start: independent positions
    positions = _insert_independent(new_feats, existing_feats, existing_embs, model, knn, gamma, distlocal)

    for it in range(max_iters):
        new_positions = np.zeros_like(positions)
        for i, feat in enumerate(new_feats):
            other_mask = np.ones(len(new_feats), dtype=bool)
            other_mask[i] = False
            feats_pool = np.vstack([existing_feats, new_feats[other_mask]])
            embs_pool = np.vstack([existing_embs, positions[other_mask]])
            new_positions[i] = _bary_insert(feat, feats_pool, embs_pool, model, knn, gamma, distlocal)

        max_shift = float(np.max(np.linalg.norm(new_positions - positions, axis=1)))
        positions = new_positions
        if max_shift < tol:
            logger.info("Iterative converged after %d rounds (shift=%.2e)", it + 1, max_shift)
            break
    else:
        logger.info("Iterative: %d rounds without full convergence", max_iters)

    return positions


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def insert_proteins(
    new_feats: np.ndarray,
    existing_feats: np.ndarray,
    existing_embs: np.ndarray,
    gamma: float = 2.0,
    knn: int = 5,
    distlocal: str = "cosine",
    strategy: str = "independent",
    max_iters: int = 15,
    dim: int = 2,
) -> tuple[np.ndarray, float]:
    """
    Insert k new proteins into an existing Poincaré map.

    Parameters
    ----------
    new_feats : (k, D)  feature vectors of the proteins to insert
    existing_feats : (N, D)  feature vectors of all proteins in the map
    existing_embs : (N, 2)  Poincaré disk coordinates of the existing proteins
    gamma : kernel bandwidth — **must match the value used at map training time**
    knn : top-k anchors used per barycenter call — match training value
    distlocal : pairwise distance metric — match training value
    strategy : one of STRATEGIES (see module docstring); default 'independent'
    max_iters : max rounds for the 'iterative' strategy
    dim : embedding dimension (2 for a standard Poincaré disk)

    Returns
    -------
    positions : (k, 2) array of new positions in the Poincaré disk
    elapsed   : wall-clock seconds for the insertion step
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES!r}, got {strategy!r}")

    new_feats = np.asarray(new_feats, dtype=np.float64)
    existing_feats = np.asarray(existing_feats, dtype=np.float64)
    existing_embs = np.asarray(existing_embs, dtype=np.float64)

    k = len(new_feats)
    logger.info("Inserting %d protein(s) with strategy=%r, gamma=%.3g, knn=%d, distlocal=%r",
                k, strategy, gamma, knn, distlocal)

    model = _make_model(dim=dim, gamma=gamma)

    t0 = time.perf_counter()

    if strategy == "independent":
        positions = _insert_independent(new_feats, existing_feats, existing_embs, model, knn, gamma, distlocal)

    elif strategy in ("seq_center_first", "seq_peri_first"):
        # Approximate insertion radius via a quick independent pass,
        # then sort by radius to determine insertion order.
        init_pos = _insert_independent(new_feats, existing_feats, existing_embs, model, knn, gamma, distlocal)
        radii = np.linalg.norm(init_pos, axis=1)
        if strategy == "seq_center_first":
            order = np.argsort(radii)          # smallest radius (most central) first
        else:
            order = np.argsort(radii)[::-1].copy()  # largest radius (most peripheral) first
        positions = _insert_sequential(new_feats, existing_feats, existing_embs, model, knn, gamma, distlocal, order)

    else:  # iterative
        positions = _insert_iterative(new_feats, existing_feats, existing_embs, model, knn, gamma, distlocal, max_iters)

    elapsed = time.perf_counter() - t0
    logger.info("Insertion complete in %.3fs", elapsed)
    return positions, elapsed
