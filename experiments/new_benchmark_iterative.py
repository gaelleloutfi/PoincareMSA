#!/usr/bin/env python3
"""
Reinsertion benchmark

- Build one full Poincaré map from features
- Sample `n_trials` indices (without replacement)
- For each index: build reduced map (full minus that index) and reinsert the point
  using four methods: bary_init, infer_bary, infer_rand, train_single
- Do NOT save coordinate CSVs for each trial. Save only:
  - the full map (interactive HTML + optional PNG)
  - a single CSV with all metrics usable for boxplots

Defaults and algorithm parameters match `experiments/new_benchmark.py`.
"""
from __future__ import annotations

import sys
import os
from pathlib import Path

# Derive project root from this file's location (experiments/ is one level below root)
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

import argparse
import time
import timeit
import numpy as np
import pandas as pd
import torch

# ensure build package on path for bare imports
build_pkg = os.path.join(project_root, 'scripts', 'build_poincare_map')
if build_pkg not in sys.path:
    sys.path.append(build_pkg)

from scripts.build_poincare_map.data import compute_rfa_w_custom_distance, prepare_data, prepare_embedding_data
from scripts.build_poincare_map.model import PoincareEmbedding, PoincareDistance
from scripts.build_poincare_map.train import train
from scripts.build_poincare_map.rsgd import RiemannianSGD

from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import kneighbors_graph
from scipy.sparse import csgraph

import plotly.express as px


def load_features(path):
    if os.path.isfile(path):
        try:
            arr = np.loadtxt(path, delimiter=',')
            return arr
        except Exception:
            df = pd.read_csv(path, header=None, index_col=None)
            return df.values
    elif os.path.isdir(path):
        try:
            feats, labels = prepare_embedding_data(path)
            return feats.numpy()
        except Exception:
            try:
                feats, labels = prepare_data(path)
                return feats.numpy()
            except Exception:
                raise ValueError(f"Cannot load features from folder: {path}")
    else:
        raise FileNotFoundError(path)


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


def quality_fn(coord_high, coord_low, k_neighbours=5):
    try:
        from scripts.build_poincare_map.embedding_quality_score import get_quality_metrics as _gqm
    except Exception:
        _gqm = None
    if _gqm is not None:
        return _gqm(coord_high=coord_high, coord_low=coord_low, distance='poincare', setting='manifold', k_neighbours=k_neighbours)
    else:
        return compute_quality_local(coord_high, coord_low, setting='manifold', k_neighbours=k_neighbours, my_metric='cosine')


def build_map_from_features(features: np.ndarray, args):
    start = timeit.default_timer()
    RFA = compute_rfa_w_custom_distance(features=features, distance_matrix=None, k_neighbours=args.knn, distfn=args.distfn, connected=args.connected, sigma=args.sigma, distlocal=args.distlocal)
    t_rfa = timeit.default_timer() - start

    indices = torch.arange(len(RFA))
    dataset = torch.utils.data.TensorDataset(indices, RFA)

    predictor = PoincareEmbedding(len(dataset), args.dim, dist=PoincareDistance, gamma=args.gamma, lossfn=args.lossfn, Qdist=args.distr, cuda=False)
    optimizer = RiemannianSGD(predictor.parameters(), lr=args.lr)

    t_start = timeit.default_timer()
    embeddings, loss, epoch = train(predictor, dataset, optimizer, args, fout=os.path.join(args.output_path, 'tmp_map'), earlystop=args.earlystop)
    t_train = timeit.default_timer() - t_start

    return np.array(embeddings), float(loss), int(epoch), float(t_rfa), float(t_train)


def main(args=None):
    if args is None:
        parser = argparse.ArgumentParser()
        parser.add_argument('--path_features', required=True)
        parser.add_argument('--output_path', required=True)
        parser.add_argument('--n_trials', type=int, default=50)
        parser.add_argument('--seed', type=int, default=42)
        parser.add_argument('--knn', type=int, default=15)
        parser.add_argument('--sigma', type=float, default=1.0)
        parser.add_argument('--gamma', type=float, default=2.0)
        parser.add_argument('--dim', type=int, default=2)
        parser.add_argument('--epochs', type=int, default=100)
        parser.add_argument('--lr', type=float, default=0.1)
        parser.add_argument('--lossfn', type=str, default='klSym')
        parser.add_argument('--distr', type=str, default='laplace')
        parser.add_argument('--distfn', type=str, default='MFIsym')
        parser.add_argument('--distlocal', type=str, default='cosine')
        parser.add_argument('--connected', type=int, default=1)
        parser.add_argument('--earlystop', type=float, default=0.0)
        args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)
    features = load_features(args.path_features)
    N = features.shape[0]
    print(f"Loaded features shape={features.shape}")

    # prepare opt for training
    class Opt:
        pass

    opt = Opt()
    opt.batchsize = max(4, int(min(128, N // 10)))
    opt.lr = args.lr
    opt.lrm = 1.0
    opt.epochs = args.epochs
    opt.burnin = max(1, args.epochs // 2)
    opt.checkout_freq = 100
    opt.debugplot = 0
    opt.earlystop = args.earlystop
    opt.seed = 42
    opt.dim = args.dim
    opt.knn = args.knn
    opt.sigma = args.sigma
    opt.gamma = args.gamma
    opt.lossfn = args.lossfn
    opt.distr = args.distr
    opt.distfn = args.distfn
    opt.distlocal = args.distlocal
    opt.connected = args.connected
    opt.output_path = args.output_path
    opt.plot = False
    opt.pca = 0

    # Build full map once
    print("Building full map...")
    emb_full, loss_full, epoch_full, t_rfa_full, t_train_full = build_map_from_features(features, opt)
    print(f"Full map: embeddings shape={emb_full.shape}, loss={loss_full:.3e}")

    try:
        Qlocal_full, Qglobal_full, _, _ = quality_fn(features, emb_full, k_neighbours=5)
    except Exception:
        Qlocal_full = Qglobal_full = float('nan')

    # save full map interactive html
    try:
        df_full_plot = pd.DataFrame(emb_full, columns=['x', 'y'])
        fig_full = px.scatter(df_full_plot, x='x', y='y', color_discrete_sequence=['lightgray'], opacity=0.8)
        fig_full.update_layout(shapes=[dict(type='circle', xref='x', yref='y', x0=-1, y0=-1, x1=1, y1=1, line_color='black')], xaxis=dict(visible=False), yaxis=dict(visible=False), width=600, height=600)
        fig_full.write_html(os.path.join(args.output_path, 'full_map.html'))
        try:
            fig_full.write_image(os.path.join(args.output_path, 'full_map.png'), scale=2)
        except Exception:
            pass
    except Exception as e:
        print('Warning: could not save full map (plotly):', e)

    rng = np.random.RandomState(args.seed)
    indices = np.arange(N)
    pool = rng.choice(indices, size=min(args.n_trials, N), replace=False)

    rows = []

    for trial_idx, idx in enumerate(pool):
        print(f"Trial {trial_idx+1}/{len(pool)}: remove index {idx}")
        mask = np.ones(N, dtype=bool)
        mask[idx] = False
        feats_rem = features[mask]

        emb_rem, loss_rem, epoch_rem, t_rfa_rem, t_train_rem = build_map_from_features(feats_rem, opt)

        # instantiate model for reduced map
        dim = emb_full.shape[1]
        model = PoincareEmbedding(size=emb_rem.shape[0], dim=dim, gamma=args.gamma, lossfn=args.lossfn, Qdist=args.distr, cuda=False)
        with torch.no_grad():
            model.lt.weight.data = torch.from_numpy(emb_rem).float()

        removed_feat = features[idx].reshape(1, -1)
        d = pairwise_distances(removed_feat, feats_rem, metric=args.distlocal).flatten()
        target = np.exp(- (model.gamma if hasattr(model, 'gamma') else args.gamma) * d)
        if target.sum() <= 0:
            target = np.ones_like(target) / float(len(target))
        else:
            target = target / target.sum()

        # prepare topk neighbors for barycenter initialization
        kb = min(5, len(target))
        topk = np.argsort(-target)[:kb]
        neighbor_embs = torch.tensor(emb_rem[topk], dtype=torch.float32)
        neighbor_w = torch.tensor(target[topk], dtype=torch.float32)
        neighbor_w = neighbor_w / neighbor_w.sum()

        # compute barycenter init (time this operation)
        try:
            t0_bary = time.perf_counter()
            x0 = model.hyperbolic_barycenter(neighbor_embs, neighbor_w, n_steps=200, tol=1e-7, alpha=1.0, device='cpu', method='karcher')
            t_bary_init = time.perf_counter() - t0_bary
            x0_np = x0.detach().cpu().numpy().reshape(-1)
        except Exception:
            x0_np = np.full(dim, np.nan)
            t_bary_init = float('nan')

        methods = {}

        # infer with barycenter init
        try:
            t0 = time.perf_counter()
            new_emb_bary = model.infer_embedding_for_point(target, n_steps=500, lr=0.05, init='barycenter', device='cpu')
            t1 = time.perf_counter() - t0
        except Exception:
            new_emb_bary = np.full(dim, np.nan)
            t1 = float('nan')
        methods['infer_bary'] = (new_emb_bary, t1)

        # infer random
        try:
            vec = np.random.normal(size=dim)
            vec = vec / (np.linalg.norm(vec) + 1e-12)
            radius = (1.0 - 1e-6) * (np.random.rand() ** (1.0 / float(dim)))
            naive_init_vec = (vec * radius).astype(float)
            t0 = time.perf_counter()
            new_emb_rand = model.infer_embedding_for_point(target, n_steps=500, lr=0.05, init='random', init_vec=naive_init_vec, device='cpu')
            t2 = time.perf_counter() - t0
        except Exception:
            new_emb_rand = np.full(dim, np.nan)
            t2 = float('nan')
        methods['infer_rand'] = (new_emb_rand, t2)

        # train single point
        try:
            t0 = time.perf_counter()
            new_emb_train, losses = model.train_single_point(target, n_steps=500, lr=0.02, init='barycenter', device='cpu', k=30, lambda_local=5.0)
            t3 = time.perf_counter() - t0
        except Exception:
            new_emb_train = np.full(dim, np.nan)
            t3 = float('nan')
        methods['train_single'] = (new_emb_train, t3)

        # include bary_init vector as an entry (record barycenter compute time)
        methods['bary_init'] = (x0_np, float(t_bary_init))

        orig_emb_vec = emb_full[idx]

        def hyperbolic_dist(a, b):
            ta = torch.tensor(a, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            tb = torch.tensor(b, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            try:
                return model.dist.apply(ta, tb).item()
            except Exception:
                return float('nan')

        for method_name, (vec, t_taken) in methods.items():
            # insert into reduced
            try:
                embs_test_from_reduced = np.insert(emb_rem, idx, np.asarray(vec).reshape(1, -1), axis=0)
            except Exception:
                embs_test_from_reduced = None

            if embs_test_from_reduced is not None:
                try:
                    Qlocal_new_reduced, Qglobal_new_reduced, _, _ = quality_fn(features, embs_test_from_reduced, k_neighbours=5)
                except Exception:
                    Qlocal_new_reduced = Qglobal_new_reduced = float('nan')
            else:
                Qlocal_new_reduced = Qglobal_new_reduced = float('nan')

            # compute Q on the canonical embs_test if available
            if embs_test_from_reduced is not None:
                try:
                    Qlocal_new, Qglobal_new, _, _ = quality_fn(features, embs_test_from_reduced, k_neighbours=5)
                except Exception:
                    Qlocal_new = Qglobal_new = float('nan')
            else:
                Qlocal_new = Qglobal_new = float('nan')

            d_to_orig = hyperbolic_dist(vec, orig_emb_vec)

            rows.append({
                'trial': int(trial_idx),
                'remove_idx': int(idx),
                'method': method_name,
                't': float(t_taken) if np.isscalar(t_taken) else float('nan'),
                'd_to_orig': float(d_to_orig),
                'Qlocal_full': float(Qlocal_full),
                'Qglobal_full': float(Qglobal_full),
                'Qlocal_new_reduced': float(Qlocal_new_reduced),
                'Qglobal_new_reduced': float(Qglobal_new_reduced),
                'Qlocal_new': float(Qlocal_new),
                'Qglobal_new': float(Qglobal_new),
            })

        # record the reduced-map build time as a separate row so we can plot map-build timings
        try:
            t_map_build = float(t_rfa_rem + t_train_rem)
        except Exception:
            t_map_build = float('nan')
        rows.append({
            'trial': int(trial_idx),
            'remove_idx': int(idx),
            'method': 'map_build',
            't': t_map_build,
            'd_to_orig': float('nan'),
            'Qlocal_full': float(Qlocal_full),
            'Qglobal_full': float(Qglobal_full),
            'Qlocal_new_reduced': float('nan'),
            'Qglobal_new_reduced': float('nan'),
            'Qlocal_new': float('nan'),
            'Qglobal_new': float('nan'),
        })

    df_out = pd.DataFrame(rows)
    out_csv = os.path.join(args.output_path, 'batch_reinsert_results.csv')
    df_out.to_csv(out_csv, index=False)
    print(f"Saved results to {out_csv}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_features', default='experiments/data/test_add_point_thioredoxins/features.csv')
    parser.add_argument('--output_path', default='experiments/batch_reinsert_out')
    parser.add_argument('--n_trials', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--knn', type=int, default=15)
    parser.add_argument('--sigma', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=2.0)
    parser.add_argument('--dim', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--lossfn', type=str, default='klSym')
    parser.add_argument('--distr', type=str, default='laplace')
    parser.add_argument('--distfn', type=str, default='MFIsym')
    parser.add_argument('--distlocal', type=str, default='cosine')
    parser.add_argument('--connected', type=int, default=1)
    parser.add_argument('--earlystop', type=float, default=0.0)
    args = parser.parse_args()
    main(args)
