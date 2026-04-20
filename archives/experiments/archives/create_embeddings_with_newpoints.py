#!/usr/bin/env python3
"""
Create CSVs containing original and inferred new points for index 0 using four methods.

For each method this script saves two files into --outdir:
 - idx0_{method}_pair.csv  -> two-row CSV with the original and the new inferred point (columns: pm1, pm2, role)
 - idx0_{method}_with_new.csv -> full embeddings CSV with the new point appended (proteins_id set to 'new_{method}')

Defaults use the kinases test dataset in `experiments/data/test_add_point_kinases`.
"""

from __future__ import annotations

import sys
# Add the project root to Python path
from pathlib import Path

# Derive project root from this file's location
# (archives/experiments/archives/ is 3 levels below root)
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.append(project_root)

import argparse
import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import pairwise_distances
from scripts.build_poincare_map.model import PoincareEmbedding


def save_pair_and_full(df_original: pd.DataFrame, orig_idx: int, new_vec: np.ndarray, method_name: str, outdir: str):
    """Save pair CSV (original + new) and full CSV with new appended."""
    os.makedirs(outdir, exist_ok=True)
    # pair df
    orig_row = df_original.iloc[orig_idx][['pm1', 'pm2']].astype(float)
    pair_df = pd.DataFrame([
        {'pm1': float(orig_row['pm1']), 'pm2': float(orig_row['pm2']), 'role': 'original'},
        {'pm1': float(new_vec[0]), 'pm2': float(new_vec[1]), 'role': method_name},
    ])
    pair_path = os.path.join(outdir, f'idx{orig_idx}_{method_name}_pair.csv')
    pair_df.to_csv(pair_path, index=False)

    # full df: original map with new appended
    full_df = df_original.copy()
    # create new row preserving columns; set proteins_id to 'new_{method}' if exists
    new_row = full_df.iloc[0].copy()
    new_row['pm1'] = float(new_vec[0])
    new_row['pm2'] = float(new_vec[1])
    # set proteins_id if column exists
    if 'proteins_id' in full_df.columns:
        new_row['proteins_id'] = f'new_{method_name}'
    new_df = pd.concat([full_df, new_row.to_frame().T], axis=0)
    full_path = os.path.join(outdir, f'idx{orig_idx}_{method_name}_with_new.csv')
    new_df.to_csv(full_path, index=False)

    print('Saved', pair_path, 'and', full_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_embedding', default='experiments/data/test_add_point_globins/PM5sigma=1.00gamma=2.00cosinepca=0_seed4.csv')
    parser.add_argument('--path_features', default='experiments/data/test_add_point_globins/features.csv')
    parser.add_argument('--idx', type=int, default=0, help='Index to remove and re-infer (default 0)')
    parser.add_argument('--outdir', default='experiments/outputs_newpoints')
    parser.add_argument('--distance_metric', default='cosine')
    parser.add_argument('--infer_steps', type=int, default=500)
    parser.add_argument('--infer_lr', type=float, default=0.05)
    parser.add_argument('--train_steps', type=int, default=500)
    parser.add_argument('--train_lr', type=float, default=0.02)
    args = parser.parse_args()

    df = pd.read_csv(args.path_embedding)
    if 'pm1' not in df.columns or 'pm2' not in df.columns:
        raise ValueError("Embedding CSV must contain 'pm1' and 'pm2')")

    embs_all = df[['pm1', 'pm2']].to_numpy(dtype=float)
    N, dim = embs_all.shape

    feats = pd.read_csv(args.path_features, index_col=0, header=None).to_numpy()

    idx = int(args.idx)
    if idx < 0 or idx >= N:
        raise IndexError('idx out of range')

    orig_emb = embs_all[idx].copy()
    orig_feat = feats[0].reshape(1, -1)  # first in features.csv as requested

    # build remaining sets
    mask = np.ones(N, dtype=bool)
    mask[idx] = False
    embs_rem = embs_all[mask]
    feats_rem = np.delete(feats, idx, axis=0)

    model = PoincareEmbedding(size=embs_rem.shape[0], dim=dim, gamma=1.0, lossfn='klSym', Qdist='laplace', cuda=False)
    with torch.no_grad():
        model.lt.weight.data = torch.from_numpy(embs_rem).float()

    # compute target from features: using first feat row
    d = pairwise_distances(orig_feat, feats_rem, metric=args.distance_metric).flatten()
    target = np.exp(- (model.gamma if hasattr(model, 'gamma') else 1.0) * d)
    if target.sum() <= 0:
        target = np.ones_like(target) / float(len(target))
    else:
        target = target / target.sum()

    # barycenter warm-start
    kb = min(5, len(target))
    topk = np.argsort(-target)[:kb]
    neighbor_embs = torch.tensor(embs_rem[topk], dtype=torch.float32)
    neighbor_w = torch.tensor(target[topk], dtype=torch.float32)
    neighbor_w = neighbor_w / neighbor_w.sum()

    # compute barycenter
    try:
        x0 = model.hyperbolic_barycenter(neighbor_embs, neighbor_w, n_steps=200, tol=1e-7, alpha=1.0, device='cpu', method='karcher')
        x0_np = x0.detach().cpu().numpy().reshape(-1)
    except Exception as e:
        print('Barycenter computation failed:', e)
        x0_np = np.full(dim, np.nan)

    # infer with barycenter init
    try:
        new_emb_bary = model.infer_embedding_for_point(target, n_steps=args.infer_steps, lr=args.infer_lr, init='barycenter', device='cpu')
    except Exception as e:
        print('infer_embedding_for_point(init=barycenter) failed:', e)
        new_emb_bary = np.full(dim, np.nan)

    # infer with random init (naive)
    try:
        max_norm = 1.0 - 1e-6
        vec = np.random.normal(size=dim)
        vec = vec / (np.linalg.norm(vec) + 1e-12)
        radius = max_norm * (np.random.rand() ** (1.0 / float(dim)))
        naive_init_vec = (vec * radius).astype(float)
        new_emb_bary2 = model.infer_embedding_for_point(target, n_steps=args.infer_steps, lr=args.infer_lr, init='random', init_vec=naive_init_vec, device='cpu')
    except Exception as e:
        print('infer_embedding_for_point(init_vec) failed:', e)
        new_emb_bary2 = np.full(dim, np.nan)

    # train_single_point refinement
    try:
        new_emb_train, losses = model.train_single_point(target, n_steps=args.train_steps, lr=args.train_lr, init='barycenter', device='cpu', k=30, lambda_local=5.0)
    except Exception as e:
        print('train_single_point failed:', e)
        new_emb_train = np.full(dim, np.nan)

    methods = [
        ('bary_init', x0_np),
        ('infer_bary', new_emb_bary),
        ('infer_initvec', new_emb_bary2),
        ('train_single', new_emb_train),
    ]

    for name, vec in methods:
        if vec is None or np.isnan(vec).any():
            print(f"Skipping {name}: invalid vector")
            continue
        save_pair_and_full(df, idx, vec, name, args.outdir)


if __name__ == '__main__':
    main()
