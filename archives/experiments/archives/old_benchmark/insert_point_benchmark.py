#!/usr/bin/env python3
"""
Reinsertion benchmark:
- For a list of indices, remove the point (features + embedding)
- Re-infer the point using several methods
- Compute hyperbolic distances between each re-inferred position and the original position
- Save the results to a CSV file

Usage (from the project root):
python3 scripts/experiments/reinsert_benchmark.py --path_embedding test_add_point_2/PM5sigma=1.00gamma=1.00cosinepca=0_seed4.csv \
    --path_features experiments/data/test_add_point_2/features.csv --out results/reinsert_benchmark.csv

"""

from __future__ import annotations

import sys

# # Add the project root to Python path
# project_root = "/home/hugo/Bureau/PoincareMSA"
# if project_root not in sys.path:
#     sys.path.append(project_root)


import argparse
import os
import numpy as np
import pandas as pd
import torch
import time
from sklearn.metrics.pairwise import pairwise_distances
from scripts.build_poincare_map.model import PoincareEmbedding
import importlib.util
import pathlib
from sklearn.neighbors import kneighbors_graph
from scipy.sparse import csgraph


def run_benchmark(path_embedding: str,
                  path_features: str,
                  out_csv: str,
                  indices: list[int] | None = None,
                  max_idx: int = 100,
                  distance_metric: str = 'cosine',
                  n_bary_k: int = 5,
                  infer_steps: int = 500,
                  infer_lr: float = 0.05,
                  train_steps: int = 500,
                  train_lr: float = 0.02,
                  train_k: int = 30,
                  train_lambda: float = 5.0,
                  device: str = 'cpu'):
    # Load data
    df = pd.read_csv(path_embedding)
    if 'pm1' not in df.columns or 'pm2' not in df.columns:
        raise ValueError('Le fichier d\'embeddings doit contenir pm1 et pm2')
    embs_all = df[['pm1', 'pm2']].to_numpy(dtype=float)
    N, dim = embs_all.shape

    # Load features (try to mimic notebook loading)
    feats_df = pd.read_csv(path_features, index_col=0, header=None)

    # ensure indices list
    if indices is None:
        indices = list(range(min(max_idx, N)))

    results = []

    # Dynamically load get_quality_metrics from the archived implementation if available
    # Prefer the local scripts implementation if present
    get_quality_metrics = None
    try:
        from scripts.build_poincare_map.embedding_quality_score import get_quality_metrics as _gqm
        get_quality_metrics = _gqm
    except Exception:
        # fallback to archived copy
        arch_path = os.path.join(os.path.dirname(__file__), '..', '..', 'archives', 'scripts', 'build_poincare_map', 'embedding_quality_score.py')
        arch_path = os.path.normpath(arch_path)
        if os.path.exists(arch_path):
            spec = importlib.util.spec_from_file_location("arch_embed_score", arch_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            if hasattr(mod, 'get_quality_metrics'):
                get_quality_metrics = mod.get_quality_metrics

    if get_quality_metrics is None:
        print("Warning: get_quality_metrics not found; Qlocal/Qglobal columns will be NaN")

    # If get_quality_metrics isn't available, provide a local fallback implementation
    def poincare_pairwise(coords: np.ndarray) -> np.ndarray:
        x = coords.astype(np.float64)
        norm2 = np.sum(x * x, axis=1)
        sqd = pairwise_distances(x, metric='euclidean') ** 2
        denom = (1.0 - norm2)[:, None] * (1.0 - norm2)[None, :]
        denom = np.maximum(denom, 1e-12)
        arg = 1.0 + 2.0 * sqd / denom
        arg = np.maximum(arg, 1.0)
        d = np.log(arg + np.sqrt((arg - 1.0) * (arg + 1.0)))
        return d

    def get_dist_manifold_local(data: np.ndarray, k_neighbours: int = 20, my_metric: str = 'cosine') -> np.ndarray:
        KNN = kneighbors_graph(data, k_neighbours, mode='distance', include_self=False, metric=my_metric).toarray()
        KNN = np.maximum(KNN, KNN.T)
        n_components, labels = csgraph.connected_components(KNN)
        if n_components > 1:
            distances = pairwise_distances(data, metric=my_metric)
            for a in range(n_components):
                idx_a = np.where(labels == a)[0]
                for b in range(a + 1, n_components):
                    idx_b = np.where(labels == b)[0]
                    sub = distances[np.ix_(idx_a, idx_b)]
                    i, j = np.unravel_index(np.argmin(sub), sub.shape)
                    KNN[idx_a[i], idx_b[j]] = distances[idx_a[i], idx_b[j]]
                    KNN[idx_b[j], idx_a[i]] = distances[idx_a[i], idx_b[j]]
        D_high = csgraph.shortest_path(KNN)
        return D_high

    def get_ranking_local(distance_matrix: np.ndarray) -> np.ndarray:
        n = distance_matrix.shape[0]
        Rank = np.zeros((n, n), dtype=int)
        for i in range(n):
            sidx = np.argsort(distance_matrix[i, :])
            idx = np.arange(n)
            Rank[i, idx[sidx][1:]] = idx[1:]
        return Rank

    def get_coRanking_local(Rank_high: np.ndarray, Rank_low: np.ndarray) -> np.ndarray:
        N = Rank_high.shape[0]
        coRank = np.zeros((N - 1, N - 1), dtype=int)
        for i in range(N):
            for j in range(N):
                k = int(Rank_high[i, j])
                l = int(Rank_low[i, j])
                if (k > 0) and (l > 0):
                    coRank[k - 1, l - 1] += 1
        return coRank

    def get_score_local(Rank_high: np.ndarray, Rank_low: np.ndarray) -> pd.DataFrame:
        coRank = get_coRanking_local(Rank_high, Rank_low)
        N = coRank.shape[0] + 1
        df_score = pd.DataFrame(columns=['Qnx', 'Bnx'])
        Qnx = 0
        Bnx = 0
        for K in range(1, N):
            Qnx += coRank[:K, K - 1].sum() + coRank[K - 1, :K].sum() - coRank[K - 1, K - 1]
            Bnx += coRank[:K, K - 1].sum() - coRank[K - 1, :K].sum()
            df_score.loc[len(df_score)] = [Qnx / (K * N), Bnx / (K * N)]
        return df_score

    def get_scalars_local(Qnx_vals: np.ndarray):
        N = len(Qnx_vals)
        K_max = 0
        val_max = Qnx_vals[0] - 1.0 / N
        for k in range(1, N):
            if val_max < (Qnx_vals[k] - (k + 1) / N):
                val_max = Qnx_vals[k] - (k + 1) / N
                K_max = k
        Qlocal = float(np.mean(Qnx_vals[:K_max + 1]))
        Qglobal = float(np.mean(Qnx_vals[K_max:]))
        return Qlocal, Qglobal, K_max

    def compute_quality_local(coord_high: np.ndarray, coord_low: np.ndarray, setting: str = 'manifold', k_neighbours: int = 20, my_metric: str = 'cosine'):
        if setting == 'global':
            D_high = pairwise_distances(coord_high, metric=my_metric)
        elif setting == 'manifold':
            D_high = get_dist_manifold_local(coord_high, k_neighbours=k_neighbours, my_metric=my_metric)
        else:
            raise NotImplementedError
        Rank_high = get_ranking_local(D_high)
        if coord_low.shape[1] == 2:
            D_low = poincare_pairwise(coord_low)
        else:
            D_low = pairwise_distances(coord_low)
        Rank_low = get_ranking_local(D_low)
        df_score = get_score_local(Rank_high, Rank_low)
        Qlocal, Qglobal, Kmax = get_scalars_local(df_score['Qnx'].values)
        return Qlocal, Qglobal, Kmax, df_score

    # helper that chooses implementation
    def quality_fn(coord_high, coord_low, k_neighbours=5):
        if get_quality_metrics is not None:
            # try to call the project's function (it expects distance/setting params)
            return get_quality_metrics(coord_high=coord_high, coord_low=coord_low, distance='poincare', setting='manifold', k_neighbours=k_neighbours)
        else:
            return compute_quality_local(coord_high, coord_low, setting='manifold', k_neighbours=k_neighbours, my_metric='cosine')

    for idx in indices:
        print(f"Processing index {idx} / {max_idx - 1}")    
        try:
            # original coordinates
            orig_emb = embs_all[idx].copy()
            # original feature row (use numpy array to avoid index-label issues)
            feats_array = feats_df.to_numpy()
            orig_feat = np.asarray(feats_array[idx]).reshape(1, -1)

            # build reduced datasets (remove index)
            mask = np.ones(N, dtype=bool)
            mask[idx] = False
            embs_rem = embs_all[mask]

            # for features, remove the row by POSITION using numpy.delete to keep shapes aligned
            feats_rem = np.delete(feats_array, idx, axis=0)

            # build model with remaining embeddings
            model = PoincareEmbedding(size=embs_rem.shape[0], dim=dim, gamma=1.0, lossfn='klSym', Qdist='laplace', cuda=False)
            with torch.no_grad():
                model.lt.weight.data = torch.from_numpy(embs_rem).float()

            # compute target from original feature -> remaining features
            d = pairwise_distances(orig_feat, feats_rem, metric=distance_metric).flatten()
            target = np.exp(- (model.gamma if hasattr(model, 'gamma') else 1.0) * d)
            if target.sum() <= 0:
                target = np.ones_like(target) / float(len(target))
            else:
                target = target / target.sum()

            # barycenter warm-start (using top-k of the target)
            kb = min(n_bary_k, len(target))
            topk = np.argsort(-target)[:kb]
            neighbor_embs = torch.tensor(embs_rem[topk], dtype=torch.float32)
            neighbor_w = torch.tensor(target[topk], dtype=torch.float32)
            neighbor_w = neighbor_w / neighbor_w.sum()

            # compute barycenter (karcher) and measure time
            try:
                t0 = time.perf_counter()
                x0 = model.hyperbolic_barycenter(neighbor_embs, neighbor_w, n_steps=200, tol=1e-7, alpha=1.0, device=device, method='karcher')
                t_bary_init = time.perf_counter() - t0
                x0_np = x0.detach().cpu().numpy().reshape(-1)
            except Exception as e:
                print('Barycenter (karcher) failed:', e)
                x0_np = np.full(dim, np.nan)
                t_bary_init = np.nan

            # method A: infer with barycenter init (uses model.hyperbolic_barycenter internally)
            try:
                t0 = time.perf_counter()
                new_emb_bary = model.infer_embedding_for_point(target, n_steps=infer_steps, lr=infer_lr, init='barycenter', device=device)
                t_infer_bary = time.perf_counter() - t0
            except Exception as e:
                print('infer_embedding_for_point(init=barycenter) failed:', e)
                new_emb_bary = np.full(dim, np.nan)
                t_infer_bary = np.nan

            # method B: infer with explicit init_vec (naive random placement in the Poincaré ball)
            try:
                # draw a random point inside the Poincaré ball (uniform radius distribution)
                max_norm = 1.0 - 1e-6
                vec = np.random.normal(size=dim)
                vec = vec / (np.linalg.norm(vec) + 1e-12)
                radius = max_norm * (np.random.rand() ** (1.0 / float(dim)))
                naive_init_vec = (vec * radius).astype(float)

                t0 = time.perf_counter()
                new_emb_bary2 = model.infer_embedding_for_point(
                    target,
                    n_steps=infer_steps,
                    lr=infer_lr,
                    init='random',
                    init_vec=naive_init_vec,
                    device=device,
                )
                t_infer_initvec = time.perf_counter() - t0
            except Exception as e:
                print('infer_embedding_for_point(init_vec) failed:', e)
                new_emb_bary2 = np.full(dim, np.nan)
                t_infer_initvec = np.nan

            # method C: refine with train_single_point (local attraction)
            try:
                t0 = time.perf_counter()
                new_emb_train, losses = model.train_single_point(target, n_steps=train_steps, lr=train_lr, init='barycenter', device=device, k=train_k, lambda_local=train_lambda)
                t_train_single = time.perf_counter() - t0
            except Exception as e:
                print('train_single_point failed:', e)
                new_emb_train = np.full(dim, np.nan)
                t_train_single = np.nan

            # diagnostics: barycenter alone distance to orig (use embs_rem topk??) For this case compute distance between x0 and original point
            # Compute Qlocal/Qglobal for original map and for the map with the new point
            Qlocal_orig = np.nan
            Qglobal_orig = np.nan
            # per-method new-map quality metrics
            Qlocal_new_baryinit = np.nan
            Qglobal_new_baryinit = np.nan
            Qlocal_new_infer_bary = np.nan
            Qglobal_new_infer_bary = np.nan
            Qlocal_new_infer_initvec = np.nan
            Qglobal_new_infer_initvec = np.nan
            Qlocal_new_train_single = np.nan
            Qglobal_new_train_single = np.nan
            try:
                # compute original quality once (we can reuse the same values for each idx)
                if 'Q_orig_cached' not in locals():
                    try:
                        Qlocal_tmp, Qglobal_tmp, _, _ = quality_fn(feats_array, embs_all, k_neighbours=5)
                        Q_orig_cached = (Qlocal_tmp, Qglobal_tmp)
                    except Exception as e:
                        print('Quality computation (original) failed:', e)
                        Q_orig_cached = (np.nan, np.nan)

                Qlocal_orig, Qglobal_orig = Q_orig_cached

                # barycenter init alone (x0_np)
                try:
                    if 'x0_np' in locals() and not np.isnan(x0_np).any():
                        embs_bary = embs_all.copy()
                        embs_bary[idx] = x0_np
                        Qlocal_new_baryinit, Qglobal_new_baryinit, _, _ = quality_fn(feats_array, embs_bary, k_neighbours=5)
                except Exception as e:
                    print('Quality computation (barycenter init) failed for idx', idx, ':', e)
                    Qlocal_new_baryinit = Qglobal_new_baryinit = np.nan

                # infer with barycenter init
                try:
                    if 'new_emb_bary' in locals() and not np.isnan(new_emb_bary).any():
                        embs_infer_bary = embs_all.copy()
                        embs_infer_bary[idx] = new_emb_bary
                        Qlocal_new_infer_bary, Qglobal_new_infer_bary, _, _ = quality_fn(feats_array, embs_infer_bary, k_neighbours=5)
                except Exception as e:
                    print('Quality computation (infer_bary) failed for idx', idx, ':', e)
                    Qlocal_new_infer_bary = Qglobal_new_infer_bary = np.nan

                # infer with random init (initvec)
                try:
                    if 'new_emb_bary2' in locals() and not np.isnan(new_emb_bary2).any():
                        embs_infer_init = embs_all.copy()
                        embs_infer_init[idx] = new_emb_bary2
                        Qlocal_new_infer_initvec, Qglobal_new_infer_initvec, _, _ = quality_fn(feats_array, embs_infer_init, k_neighbours=5)
                except Exception as e:
                    print('Quality computation (infer_initvec) failed for idx', idx, ':', e)
                    Qlocal_new_infer_initvec = Qglobal_new_infer_initvec = np.nan

                # train_single result
                try:
                    if 'new_emb_train' in locals() and not np.isnan(new_emb_train).any():
                        embs_train = embs_all.copy()
                        embs_train[idx] = new_emb_train
                        Qlocal_new_train_single, Qglobal_new_train_single, _, _ = quality_fn(feats_array, embs_train, k_neighbours=5)
                except Exception as e:
                    print('Quality computation (train_single) failed for idx', idx, ':', e)
                    Qlocal_new_train_single = Qglobal_new_train_single = np.nan
            except Exception as e:
                print('Q metrics computation failed:', e)
                Qlocal_orig = Qglobal_orig = Qlocal_new = Qglobal_new = np.nan

            try:
                orig_t = torch.tensor(orig_emb, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                def to_tensor(v):
                    return torch.tensor(v, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

                d_infer_bary = float('nan')
                d_infer_bary2 = float('nan')
                d_train_single = float('nan')
                d_bary_init = float('nan')

                if not np.isnan(new_emb_bary).any():
                    d_infer_bary = model.dist.apply(to_tensor(new_emb_bary), orig_t).item()
                if not np.isnan(new_emb_bary2).any():
                    d_infer_bary2 = model.dist.apply(to_tensor(new_emb_bary2), orig_t).item()
                if not np.isnan(new_emb_train).any():
                    d_train_single = model.dist.apply(to_tensor(new_emb_train), orig_t).item()
                if not np.isnan(x0_np).any():
                    d_bary_init = model.dist.apply(to_tensor(x0_np), orig_t).item()

            except Exception as e:
                print('Distance computation failed:', e)
                d_infer_bary = d_infer_bary2 = d_train_single = d_bary_init = np.nan

            results.append({
                'idx': idx,
                'd_infer_bary': d_infer_bary,
                'd_infer_initvec': d_infer_bary2,
                'd_train_single': d_train_single,
                'd_bary_init': d_bary_init,
                'Qlocal_orig': Qlocal_orig,
                'Qglobal_orig': Qglobal_orig,
                'Qlocal_new_baryinit': Qlocal_new_baryinit,
                'Qglobal_new_baryinit': Qglobal_new_baryinit,
                'Qlocal_new_infer_bary': Qlocal_new_infer_bary,
                'Qglobal_new_infer_bary': Qglobal_new_infer_bary,
                'Qlocal_new_infer_initvec': Qlocal_new_infer_initvec,
                'Qglobal_new_infer_initvec': Qglobal_new_infer_initvec,
                'Qlocal_new_train_single': Qlocal_new_train_single,
                'Qglobal_new_train_single': Qglobal_new_train_single,
                't_bary_init': t_bary_init,
                't_infer_bary': t_infer_bary,
                't_infer_initvec': t_infer_initvec,
                't_train_single': t_train_single,
            })

        except Exception as e_outer:
            print(f'Failed for idx {idx}:', e_outer)
            results.append({
                'idx': idx,
                'd_infer_bary': np.nan,
                'd_infer_initvec': np.nan,
                'd_train_single': np.nan,
                'd_bary_init': np.nan,
            })

    df_res = pd.DataFrame(results)
    os.makedirs(os.path.dirname(out_csv) or '.', exist_ok=True)
    df_res.to_csv(out_csv, index=False)
    print('Saved results to', out_csv)
    return df_res


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_embedding', default='experiments/data/test_add_point_thioredoxins/PM5sigma=1.00gamma=2.00cosinepca=0_seed4.csv',
                        help='Chemin vers le fichier d\'embeddings (csv)')
    parser.add_argument('--path_features', default='experiments/data/test_add_point_thioredoxins/features.csv',
                        help='Chemin vers le fichier de features (csv)')
    parser.add_argument('--out', default='experiments/results/results_thioredoxins/reinsert_benchmark.csv',
                        help='Chemin du fichier CSV de sortie')
    parser.add_argument('--max_idx', type=int, default=100, help='Nombre maximal d\'indices à tester')
    parser.add_argument('--device', default='cpu', help='Device to run on (cpu or cuda)')
    args = parser.parse_args()

    # If run from an editor like VSCode (Run Python File) the defaults above
    # allow immediate execution without CLI args. They can still be overridden
    # when calling the script from a terminal.
    run_benchmark(args.path_embedding, args.path_features, args.out, max_idx=args.max_idx, device=args.device)
