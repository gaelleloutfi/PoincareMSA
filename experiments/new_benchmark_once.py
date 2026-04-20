#!/usr/bin/env python3
"""
New benchmark script:
- Build a Poincaré map from a features CSV (or folder of embeddings)
- Compute Qlocal/Qglobal and runtime for the full map
- Remove one point (by index), rebuild the map on reduced features
- Reinsert the removed point using four methods and compare distances, Q and timings

Saves results to `output_dir/results.csv` and prints a short summary.

This script reuses utilities from `scripts/build_poincare_map`.
"""
from __future__ import annotations

# Add the project root to Python path
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

# Ensure local build_poincare_map directory is on sys.path so modules that
# use bare imports (e.g. `from poincare_maps import ...`) resolve when the
# script is run without setting PYTHONPATH externally.
build_pkg = os.path.join(project_root, 'scripts', 'build_poincare_map')
if build_pkg not in sys.path:
    sys.path.append(build_pkg)

from scripts.build_poincare_map.data import compute_rfa_w_custom_distance, prepare_data, prepare_embedding_data
from scripts.build_poincare_map.model import PoincareEmbedding, PoincareDistance
from scripts.build_poincare_map.train import train
from scripts.build_poincare_map.rsgd import RiemannianSGD

# plotting helper from project
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import kneighbors_graph
from scipy.sparse import csgraph


def load_features(path):
    # If path is a file (csv or txt), try to load numeric matrix
    if os.path.isfile(path):
        try:
            arr = np.loadtxt(path, delimiter=',')
            return arr
        except Exception:
            # try pandas fallback
            df = pd.read_csv(path, header=None, index_col=None)
            return df.values
    elif os.path.isdir(path):
        # try prepare_embedding_data (.pt) then prepare_data (.aamtx)
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
    # features: numpy array (N, d)
    # compute RFA
    start = timeit.default_timer()
    RFA = compute_rfa_w_custom_distance(features=features, distance_matrix=None, k_neighbours=args.knn, distfn=args.distfn, connected=args.connected, sigma=args.sigma, distlocal=args.distlocal)
    t_rfa = timeit.default_timer() - start

    # dataset and model
    indices = torch.arange(len(RFA))
    dataset = torch.utils.data.TensorDataset(indices, RFA)

    predictor = PoincareEmbedding(len(dataset), args.dim, dist=PoincareDistance, gamma=args.gamma, lossfn=args.lossfn, Qdist=args.distr, cuda=False)
    # The PoincareEmbedding constructor in model expects some args; to be robust, construct with common signature
    # instantiate optimizer
    optimizer = RiemannianSGD(predictor.parameters(), lr=args.lr)

    t_start = timeit.default_timer()
    embeddings, loss, epoch = train(predictor, dataset, optimizer, args, fout=os.path.join(args.output_path, 'tmp_map'), earlystop=args.earlystop)
    t_train = timeit.default_timer() - t_start

    return np.array(embeddings), float(loss), int(epoch), float(t_rfa), float(t_train)


def main(args=None):
    # If args is None, parse from command line. Otherwise `args` should be an
    # argparse.Namespace (useful for programmatic calls or to edit defaults
    # below in the __main__ block).
    if args is None:
        parser = argparse.ArgumentParser()
        parser.add_argument('--path_features', required=True, help='Path to features CSV or embedding folder')
        parser.add_argument('--output_path', required=True, help='Folder to store results')
        parser.add_argument('--remove_idx', type=int, default=0, help='Index of the point to remove and reinsert')
        parser.add_argument('--knn', type=int, default=15)
        parser.add_argument('--sigma', type=float, default=1.0)
        parser.add_argument('--gamma', type=float, default=2.0)
        parser.add_argument('--dim', type=int, default=2)
        parser.add_argument('--epochs', type=int, default=400)
        parser.add_argument('--lr', type=float, default=0.1)
        parser.add_argument('--lossfn', type=str, default='klSym')
        parser.add_argument('--distr', type=str, default='laplace')
        parser.add_argument('--distfn', type=str, default='MFIsym')
        parser.add_argument('--distlocal', type=str, default='cosine')
        parser.add_argument('--connected', type=int, default=1)
        parser.add_argument('--earlystop', type=float, default=0.0)
        args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)
    # load features
    features = load_features(args.path_features)
    N = features.shape[0]
    print(f"Loaded features shape={features.shape}")

    # prepare args object for train() compatibility
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

    # Build full map
    print("Building full map...")
    emb_full, loss_full, epoch_full, t_rfa_full, t_train_full = build_map_from_features(features, opt)
    print(f"Full map: embeddings shape={emb_full.shape}, loss={loss_full:.3e}, rfa_time={t_rfa_full:.2f}s, train_time={t_train_full:.2f}s")

    # Compute original quality
    try:
        Qlocal_full, Qglobal_full, _, _ = quality_fn(features, emb_full, k_neighbours=5)
    except Exception:
        Qlocal_full = Qglobal_full = float('nan')

    # Now remove the specified index and build reduced map
    idx = args.remove_idx
    if idx < 0 or idx >= N:
        raise IndexError('remove_idx out of range')

    mask = np.ones(N, dtype=bool)
    mask[idx] = False
    feats_rem = features[mask]

    print(f"Building reduced map without index {idx} (N -> {feats_rem.shape[0]})...")
    emb_rem, loss_rem, epoch_rem, t_rfa_rem, t_train_rem = build_map_from_features(feats_rem, opt)
    print(f"Reduced map built: embeddings shape={emb_rem.shape}")

    # Save the original full embeddings and a simple full map (all points gray)
    try:
        df_full = pd.DataFrame(emb_full, columns=['pm1', 'pm2'])
        full_path = os.path.join(args.output_path, 'embeddings_full.csv')
        df_full.to_csv(full_path, index=False)
        # simple grayscale interactive plot for the full map (Plotly)
        try:
            df_full_plot = pd.DataFrame(emb_full, columns=['x', 'y'])
            fig_full = px.scatter(df_full_plot, x='x', y='y', color_discrete_sequence=['lightgray'], opacity=0.8)
            # add unit circle
            fig_full.update_layout(shapes=[dict(type='circle', xref='x', yref='y', x0=-1, y0=-1, x1=1, y1=1, line_color='black')])
            fig_full.update_layout(xaxis=dict(showgrid=False, visible=False), yaxis=dict(showgrid=False, visible=False), width=600, height=600)
            full_html = os.path.join(args.output_path, 'full_map.html')
            fig_full.write_html(full_html)
            # try static export if available
            try:
                fig_full.write_image(os.path.join(args.output_path, 'full_map.png'), scale=2)
            except Exception:
                pass
        except Exception as e:
            print('Warning: could not save/plot full map (plotly):', e)
    except Exception as e:
        print('Warning: could not save/plot full map:', e)

    # Save reduced embeddings and a simple reduced map (all points gray)
    try:
        df_rem = pd.DataFrame(emb_rem, columns=['pm1', 'pm2'])
        rem_path = os.path.join(args.output_path, 'embeddings_reduced.csv')
        df_rem.to_csv(rem_path, index=False)
        try:
            df_rem_plot = pd.DataFrame(emb_rem, columns=['x', 'y'])
            fig_rem = px.scatter(df_rem_plot, x='x', y='y', color_discrete_sequence=['lightgray'], opacity=0.8)
            fig_rem.update_layout(shapes=[dict(type='circle', xref='x', yref='y', x0=-1, y0=-1, x1=1, y1=1, line_color='black')])
            fig_rem.update_layout(xaxis=dict(showgrid=False, visible=False), yaxis=dict(showgrid=False, visible=False), width=600, height=600)
            rem_html = os.path.join(args.output_path, 'reduced_map.html')
            fig_rem.write_html(rem_html)
            try:
                fig_rem.write_image(os.path.join(args.output_path, 'reduced_map.png'), scale=2)
            except Exception:
                pass
        except Exception as e:
            print('Warning: could not save/plot reduced map (plotly):', e)
    except Exception as e:
        print('Warning: could not save/plot reduced map:', e)

    # instantiate a PoincareEmbedding object representing the reduced map
    dim = emb_full.shape[1]
    model = PoincareEmbedding(size=emb_rem.shape[0], dim=dim, gamma=args.gamma, lossfn=args.lossfn, Qdist=args.distr, cuda=False)
    with torch.no_grad():
        model.lt.weight.data = torch.from_numpy(emb_rem).float()

    # Prepare target vector for the removed point (distances from removed feature to remaining features)
    removed_feat = features[idx].reshape(1, -1)
    d = pairwise_distances(removed_feat, feats_rem, metric=args.distlocal).flatten()
    target = np.exp(- (model.gamma if hasattr(model, 'gamma') else args.gamma) * d)
    if target.sum() <= 0:
        target = np.ones_like(target) / float(len(target))
    else:
        target = target / target.sum()

    results = {}

    # barycenter warm-start (top-k)
    kb = min(5, len(target))
    topk = np.argsort(-target)[:kb]
    neighbor_embs = torch.tensor(emb_rem[topk], dtype=torch.float32)
    neighbor_w = torch.tensor(target[topk], dtype=torch.float32)
    neighbor_w = neighbor_w / neighbor_w.sum()

    try:
        t0 = time.perf_counter()
        x0 = model.hyperbolic_barycenter(neighbor_embs, neighbor_w, n_steps=200, tol=1e-7, alpha=1.0, device='cpu', method='karcher')
        t_bary_init = time.perf_counter() - t0
        x0_np = x0.detach().cpu().numpy().reshape(-1)
    except Exception as e:
        print('Barycenter failed:', e)
        x0_np = np.full(dim, np.nan)
        t_bary_init = float('nan')

    results['bary_init'] = {'vec': x0_np, 't': t_bary_init}

    # Method 1: infer with barycenter init
    try:
        t0 = time.perf_counter()
        new_emb_bary = model.infer_embedding_for_point(target, n_steps=500, lr=0.05, init='barycenter', device='cpu')
        t1 = time.perf_counter() - t0
    except Exception as e:
        print('infer(init=barycenter) failed:', e)
        new_emb_bary = np.full(dim, np.nan)
        t1 = float('nan')
    results['infer_bary'] = {'vec': new_emb_bary, 't': t1}

    # Method 2: infer with random init_vec
    try:
        vec = np.random.normal(size=dim)
        vec = vec / (np.linalg.norm(vec) + 1e-12)
        radius = (1.0 - 1e-6) * (np.random.rand() ** (1.0 / float(dim)))
        naive_init_vec = (vec * radius).astype(float)
        t0 = time.perf_counter()
        new_emb_rand = model.infer_embedding_for_point(target, n_steps=500, lr=0.05, init='random', init_vec=naive_init_vec, device='cpu')
        t2 = time.perf_counter() - t0
    except Exception as e:
        print('infer(init_vec) failed:', e)
        new_emb_rand = np.full(dim, np.nan)
        t2 = float('nan')
    results['infer_rand'] = {'vec': new_emb_rand, 't': t2}

    # Method 3: train_single_point (local refinement)
    try:
        t0 = time.perf_counter()
        new_emb_train, losses = model.train_single_point(target, n_steps=500, lr=0.02, init='barycenter', device='cpu', k=30, lambda_local=5.0)
        t3 = time.perf_counter() - t0
    except Exception as e:
        print('train_single_point failed:', e)
        new_emb_train = np.full(dim, np.nan)
        t3 = float('nan')
    results['train_single'] = {'vec': new_emb_train, 't': t3}

    # Diagnostics: distances to original embedding (if available)
    orig_emb_vec = emb_full[idx]

    def hyperbolic_dist(a, b):
        ta = torch.tensor(a, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        tb = torch.tensor(b, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        try:
            return model.dist.apply(ta, tb).item()
        except Exception:
            return float('nan')

    out_rows = []
    for name, info in results.items():
        vec = info['vec']
        t_taken = info['t']
        d_to_orig = hyperbolic_dist(vec, orig_emb_vec)
        # Variant: insert into reduced embedding so other points equal emb_rem
        Qlocal_new_reduced = Qglobal_new_reduced = float('nan')
        embs_test_from_reduced = None
        try:
            embs_test_from_reduced = np.insert(emb_rem, idx, np.asarray(vec).reshape(1, -1), axis=0)
            df_test2 = pd.DataFrame(embs_test_from_reduced, columns=['pm1', 'pm2'])
            emb_name2 = f'embeddings_{name}_from_reduced.csv'
            df_test2.to_csv(os.path.join(args.output_path, emb_name2), index=False)
            try:
                Qlocal_new_reduced, Qglobal_new_reduced, _, _ = quality_fn(features, embs_test_from_reduced, k_neighbours=5)
            except Exception:
                Qlocal_new_reduced = Qglobal_new_reduced = float('nan')
        except Exception as e:
            print(f'Warning (from_reduced) for {name}:', e)

        # Per-method interactive plot (Plotly): other points gray, neighbors orange, added point blue
        try:
            df_base = pd.DataFrame(emb_rem, columns=['x', 'y'])
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_base['x'], y=df_base['y'], mode='markers', marker=dict(color='lightgray', size=6), name='other'))
            # neighbors
            try:
                df_neigh = pd.DataFrame(emb_rem[topk], columns=['x', 'y'])
                fig.add_trace(go.Scatter(x=df_neigh['x'], y=df_neigh['y'], mode='markers', marker=dict(color='orange', size=8), name='neighbors'))
            except Exception:
                pass
            # added point
            if embs_test_from_reduced is not None:
                newpt = np.asarray(vec).reshape(-1)
                fig.add_trace(go.Scatter(x=[newpt[0]], y=[newpt[1]], mode='markers', marker=dict(color='blue', size=12), name='added'))
            fig.update_layout(shapes=[dict(type='circle', xref='x', yref='y', x0=-1, y0=-1, x1=1, y1=1, line_color='black')], xaxis=dict(visible=False), yaxis=dict(visible=False), width=600, height=600)
            html_path = os.path.join(args.output_path, f'map_{name}_from_reduced.html')
            fig.write_html(html_path)
            try:
                fig.write_image(os.path.join(args.output_path, f'map_{name}_from_reduced.png'), scale=2)
            except Exception:
                pass
        except Exception as e:
            print(f'Warning: could not create method plot for {name} (plotly):', e)

        # --- Canonical choice: use the embedding reconstructed from the reduced map
        if embs_test_from_reduced is not None:
            embs_test = embs_test_from_reduced
        else:
            embs_test = None

        # Save canonical embeddings and plot (reduced-based reconstruction)
        if embs_test is not None:
            try:
                df_canon = pd.DataFrame(embs_test, columns=['pm1', 'pm2'])
                emb_name_canon = f'embeddings_{name}.csv'
                df_canon.to_csv(os.path.join(args.output_path, emb_name_canon), index=False)
            except Exception as e:
                print(f'Warning: could not save canonical embeddings/plot for {name}:', e)
            try:
                Qlocal_new, Qglobal_new, _, _ = quality_fn(features, embs_test, k_neighbours=5)
            except Exception:
                Qlocal_new = Qglobal_new = float('nan')
        else:
            Qlocal_new = Qglobal_new = float('nan')

        out_rows.append({
            'method': name,
            't': t_taken,
            'd_to_orig': d_to_orig,
            'Qlocal_full': Qlocal_full,
            'Qglobal_full': Qglobal_full,
            'Qlocal_new_reduced': Qlocal_new_reduced,
            'Qglobal_new_reduced': Qglobal_new_reduced,
            'Qlocal_new': Qlocal_new,
            'Qglobal_new': Qglobal_new,
        })

    df_out = pd.DataFrame(out_rows)
    out_csv = os.path.join(args.output_path, 'new_benchmark_results.csv')
    df_out.to_csv(out_csv, index=False)
    print(f"Saved results to {out_csv}")
    print(df_out)


if __name__ == '__main__':
    # Editable defaults for quick runs from an editor. Change these values as
    # needed and re-run the script. They are only used when running the file
    # directly (not when importing `main` from other modules).
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_features', default='experiments/data/test_add_point_kinases/features.csv', help='Path to features CSV or embedding folder')
    parser.add_argument('--output_path', default='experiments/new_benchmark_out', help='Folder to store results')
    parser.add_argument('--remove_idx', type=int, default=0, help='Index of the point to remove and reinsert')
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

    # Call main with the Namespace so defaults are easy to tweak here.
    main(args)
