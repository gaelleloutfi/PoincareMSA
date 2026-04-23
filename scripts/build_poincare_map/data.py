# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""Utilities to prepare data and compute RFA (random forest affinity) matrices.

This module contains routines to:
 - construct tensors from PSSM or embedding files,
 - prepare feature matrices and labels,
 - compute KNN, similarity S, Laplacian and the RFA matrix.

Only a subset of the original repository functionality is required by the
main pipeline; the code here keeps the public functions used by
`scripts/build_poincare_map/main.py` and documents their behavior.
"""

from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import pairwise_distances
from scipy.sparse import csgraph

import numpy as np
import pandas as pd
import torch
import os
import timeit
import logging

# Module-level logger
logger = logging.getLogger(__name__)


def _robust_load_csv_matrix(path):
    """Load a numeric CSV as a pandas DataFrame robustly.

    Behavior:
    - Prefer ``header=None`` so numeric CSVs without headers are read as data.
    - Fall back to ``header=0`` if that fails (file might have a header row).
    - Final fallback to ``numpy.loadtxt`` for very odd formats.
    - If the resulting DataFrame has shape (n, n+1) it will detect a likely
      index/label first column (e.g. produced by pandas.to_csv with index)
      and drop it when appropriate.
    """
    if path is None:
        raise ValueError("path must be provided")
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    # Try the simplest, safest read first (no header)
    try:
        df = pd.read_csv(path, header=None)
    except Exception:
        # If that fails, try reading with a header row
        try:
            df = pd.read_csv(path, header=0)
        except Exception:
            # Last resort: numpy
            arr = np.loadtxt(path, delimiter=',')
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            df = pd.DataFrame(arr)

    # Detect and handle two common issues that produce shape (n, n+1):
    # 1) The CSV had a header row that contains numeric values (the header was
    #    incorrectly parsed as column names). In that case the true matrix has
    #    an extra top row which pandas interpreted as header. We recover by
    #    treating the header values as the first data row.
    # 2) The CSV contains a left-most index/ID column (e.g. 'Unnamed: 0' or
    #    integer sequence) which should be dropped.
    r, c = df.shape
    if c == r + 1:
        # Attempt 1: pandas parsed the first CSV line as header. Read the raw
        # first line and try to parse it as numeric; if its length matches the
        # number of columns, prepend it to the DataFrame values. If this
        # produces a square matrix, adopt it.
        try:
            with open(path, 'r') as fh:
                first_line = fh.readline().strip('\n')
            first_vals = [s.strip() for s in first_line.split(',')]
            first_floats = None
            try:
                first_floats = [float(s) for s in first_vals]
            except Exception:
                first_floats = None

            if first_floats is not None and len(first_floats) == c:
                try:
                    remaining = df.values.astype(float)
                    new_arr = np.vstack([np.array(first_floats, dtype=float), remaining])
                    if new_arr.shape[0] == new_arr.shape[1]:
                        df = pd.DataFrame(new_arr)
                        r, c = df.shape
                except Exception:
                    # if conversion fails, fall through to heuristics below
                    pass
        except Exception:
            pass

        # Attempt 2: if column names themselves look numeric, treat them as a
        # header row and prepend them to the data values (fallback).
        try:
            colnums = pd.to_numeric(df.columns, errors='coerce')
            num_colnums = int(colnums.notnull().sum())
        except Exception:
            num_colnums = 0

        if num_colnums >= max(3, int(0.9 * c)):
            try:
                header_as_row = colnums.values.astype(float)
                remaining = df.values.astype(float)
                new_arr = np.vstack([header_as_row, remaining])
                df = pd.DataFrame(new_arr)
                r, c = df.shape
            except Exception:
                pass

        # Attempt 3: if still shape (n, n+1), the left-most column may be an
        # index/ID column -> drop it when heuristics indicate it's an index.
        if df.shape[1] == df.shape[0] + 1:
            first_col = df.iloc[:, 0]
            drop_first = False
            try:
                coerced = pd.to_numeric(first_col, errors='coerce')
                num_numeric = int(coerced.notnull().sum())
                # If very few entries are numeric, it's probably an index/ID
                if num_numeric < max(1, int(0.1 * df.shape[0])):
                    drop_first = True
                else:
                    vals = first_col.astype(str).str.strip()
                    if vals.apply(lambda v: v.isdigit()).all():
                        ints = vals.astype(int)
                        if (np.array_equal(ints, np.arange(df.shape[0])) or
                                np.array_equal(ints, np.arange(1, df.shape[0] + 1))):
                            drop_first = True
            except Exception:
                drop_first = True

            first_col_name = df.columns[0]
            if isinstance(first_col_name, str) and 'Unnamed' in first_col_name:
                drop_first = True

            if drop_first:
                df = df.iloc[:, 1:].reset_index(drop=True)

    return df


def append_point_to_feature_and_distance(
    features_path,
    distance_path,
    new_feature,
    new_id=None,
    labels_path=None,
    out_features_path=None,
    out_distance_path=None,
    out_labels_path=None,
    metric='euclidean',
):
    """Append a new feature vector to `features_path` and update `distance_path`.

    Parameters
    - features_path: path to CSV (numeric) with shape (n, d)
    - distance_path: path to CSV (numeric) with shape (n, n)
    - new_feature: 1D array-like or path to a CSV/np file containing the new vector
    - new_id: optional identifier for the new row (used only if labels_path provided)
    - labels_path: optional path to labels CSV to append the new_id
    - out_*: optional output paths; if None, original files are overwritten
    - metric: distance metric passed to sklearn.metrics.pairwise_distances

    Returns a dict with keys: features_path, distance_path, labels_path, shapes
    """
    # Load features
    df_feat = _robust_load_csv_matrix(features_path)
    X = df_feat.values.astype(float)

    # Load or parse new_feature
    if isinstance(new_feature, str):
        # try loading numeric file
        if os.path.exists(new_feature):
            try:
                nf = np.loadtxt(new_feature, delimiter=',')
            except Exception:
                nf = pd.read_csv(new_feature, header=None).values.reshape(-1)
        else:
            raise FileNotFoundError(f"new_feature path does not exist: {new_feature}")
    else:
        nf = np.asarray(new_feature, dtype=float)

    nf = nf.reshape(-1)
    if nf.shape[0] != X.shape[1]:
        raise ValueError(f"New feature dimensionality ({nf.shape[0]}) does not match features ({X.shape[1]})")

    # Append feature
    df_new_row = pd.DataFrame([nf])
    df_feat_app = pd.concat([df_feat.reset_index(drop=True), df_new_row], ignore_index=True)

    # Load distance matrix
    df_dist = _robust_load_csv_matrix(distance_path)
    D = df_dist.values.astype(float)
    if D.shape[0] != D.shape[1]:
        raise ValueError(f"distance matrix must be square, got shape {D.shape}")
    if D.shape[0] != X.shape[0]:
        raise ValueError(f"distance matrix size ({D.shape[0]}) does not match number of features ({X.shape[0]})")

    # Compute distances from new point to existing features
    dists = pairwise_distances(X, nf.reshape(1, -1), metric=metric).reshape(-1)

    # Build new distance matrix (add new row and new column)
    new_col = dists.reshape(-1, 1)
    # new row is same as new_col transposed, with zero at diagonal
    new_row = np.concatenate([dists, np.array([0.0])])

    D_app = np.vstack([np.hstack([D, new_col]), new_row.reshape(1, -1)])

    # Prepare output paths
    out_fpath = out_features_path or features_path
    out_dpath = out_distance_path or distance_path

    # Save updated features and distance matrix
    np.savetxt(out_fpath, df_feat_app.values, delimiter=',')
    np.savetxt(out_dpath, D_app, delimiter=',')
    logger.info("Wrote features to %s (shape=%s)", out_fpath, df_feat_app.shape)
    logger.info("Wrote distance matrix to %s (shape=%s)", out_dpath, D_app.shape)

    out_info = {
        'features_path': out_fpath,
        'distance_path': out_dpath,
        'features_shape': df_feat_app.shape,
        'distance_shape': D_app.shape,
    }

    # Handle labels
    if labels_path is not None:
        try:
            df_labels = pd.read_csv(labels_path, header=None)
        except Exception:
            df_labels = pd.DataFrame(np.loadtxt(labels_path, dtype=str, delimiter=','))

        # append new id or empty string
        new_label = new_id if new_id is not None else ''
        df_labels_app = pd.concat([df_labels.reset_index(drop=True), pd.DataFrame([ [new_label] ])], ignore_index=True)
        out_lpath = out_labels_path or labels_path
        df_labels_app.to_csv(out_lpath, index=False, header=False)
        logger.info("Wrote labels to %s (shape=%s)", out_lpath, df_labels_app.shape)
        out_info['labels_path'] = out_lpath
        out_info['labels_shape'] = df_labels_app.shape

    return out_info


def find_k_neighbors_for_new_point(features_path, new_feature, k=5, labels_path=None, metric=None, distlocal=None):
    """Find the k nearest neighbors (indices and distances) of a new feature
    vector among the existing features stored in `features_path`.

    This function now tries to use the same distance metric handling as the
    main pipeline: prefer the ``distlocal`` argument (used by the main
    computation), otherwise fall back to the legacy ``metric`` parameter.

    Parameters
    - features_path: path to CSV with shape (n, d)
    - new_feature: 1D array-like or path to a CSV/np file containing the new vector
    - k: number of neighbors to return
    - labels_path: optional path to a labels CSV (one id per row). If provided,
      the returned neighbors will include the corresponding protein_id.
    - metric: legacy name for distance metric passed to
      sklearn.metrics.pairwise_distances (kept for backward compatibility)
    - distlocal: preferred name matching the CLI/main pipeline (e.g. 'minkowski'
      or 'cosine'). If provided, this will be used in preference to ``metric``.

    Returns a list of dicts: [{'index': int, 'distance': float, 'protein_id': str|None}, ...]
    """
    # Load features
    df_feat = _robust_load_csv_matrix(features_path)
    X = df_feat.values.astype(float)

    # Load or parse new_feature
    if isinstance(new_feature, str):
        if os.path.exists(new_feature):
            try:
                nf = np.loadtxt(new_feature, delimiter=',')
            except Exception:
                nf = pd.read_csv(new_feature, header=None).values.reshape(-1)
        else:
            raise FileNotFoundError(f"new_feature path does not exist: {new_feature}")
    else:
        nf = np.asarray(new_feature, dtype=float)

    nf = nf.reshape(-1)
    if nf.shape[0] != X.shape[1]:
        raise ValueError(f"New feature dimensionality ({nf.shape[0]}) does not match features ({X.shape[1]})")

    # Decide which metric to use: prefer distlocal (pipeline name), then metric,
    # then default to 'euclidean'
    chosen_metric = distlocal if distlocal is not None else (metric if metric is not None else 'euclidean')

    # Compute distances using sklearn's pairwise_distances which is compatible
    # with the metrics used by kneighbors_graph in the main pipeline.
    # This ensures the same behaviour for 'minkowski' or 'cosine'.
    dists = pairwise_distances(X, nf.reshape(1, -1), metric=chosen_metric).reshape(-1)
    idx_sorted = np.argsort(dists)[:k]

    # Load labels if available
    protein_ids = None
    if labels_path is not None and os.path.exists(labels_path):
        try:
            df_labels = pd.read_csv(labels_path, header=None)
            # flatten to list
            if df_labels.shape[1] == 1:
                protein_ids = df_labels.iloc[:, 0].astype(str).tolist()
            else:
                # if multiple columns, join them with '|'
                protein_ids = df_labels.astype(str).agg('|'.join, axis=1).tolist()
        except Exception:
            # fallback to numpy load
            try:
                arr = np.loadtxt(labels_path, dtype=str, delimiter=',')
                protein_ids = arr.reshape(-1).astype(str).tolist()
            except Exception:
                protein_ids = None

    neighbors = []
    for i in idx_sorted:
        pid = protein_ids[i] if (protein_ids is not None and i < len(protein_ids)) else None
        neighbors.append({'index': int(i), 'distance': float(dists[i]), 'protein_id': pid})

    return neighbors


def append_centroid_to_embedding(embedding_path, features_path, new_feature,
                                 k=5, labels_path=None, new_id=None,
                                 out_embedding_path=None, metric='euclidean'):
    """Append a new point to a 2D embedding CSV by computing the centroid of
    the k nearest neighbors of `new_feature` in feature space.

    Parameters
    - embedding_path: CSV path containing columns ['pm1','pm2'] and optionally 'proteins_id'
    - features_path: path to numeric features CSV used to compute neighbors
    - new_feature: 1D array-like or path to file containing the new vector
    - k: number of neighbors to use for centroid
    - labels_path: optional labels CSV path (used by neighbor lookup)
    - new_id: identifier to store in 'proteins_id' for the new row
    - out_embedding_path: where to write the appended embedding (defaults to embedding_path with .appended.csv)
    - metric: distance metric for neighbor search

    Returns a tuple (appended_df, neighbors) where neighbors is the list returned
    by `find_k_neighbors_for_new_point`.
    """
    # Compute neighbors
    neighbors = find_k_neighbors_for_new_point(features_path, new_feature, k=k, labels_path=labels_path, metric=metric)

    # Load embedding
    if not os.path.exists(embedding_path):
        raise FileNotFoundError(f"Embedding file not found: {embedding_path}")
    df_emb = pd.read_csv(embedding_path, header=0)

    # Ensure pm1/pm2 exist
    if not all(c in df_emb.columns for c in ['pm1', 'pm2']):
        raise ValueError("Embedding file must contain 'pm1' and 'pm2' columns")

    # Collect neighbor coordinates
    coords = []
    for n in neighbors:
        idx = n.get('index')
        pid = n.get('protein_id')
        row = None
        # Prefer index-based lookup (fast and direct) if within bounds
        try:
            if idx is not None and 0 <= idx < len(df_emb):
                row = df_emb.iloc[idx]
        except Exception:
            row = None

        # Fallback: try matching by proteins_id column if present and pid available
        if row is None and pid is not None and 'proteins_id' in df_emb.columns:
            matches = df_emb[df_emb['proteins_id'].astype(str) == str(pid)]
            if len(matches) > 0:
                row = matches.iloc[0]

        if row is None:
            # neighbor could not be resolved in embedding; raise informative error
            raise ValueError(f"Could not locate neighbor {n} in embedding {embedding_path}")

        coords.append((float(row['pm1']), float(row['pm2'])))

    # Compute centroid
    arr = np.array(coords)
    centroid = arr.mean(axis=0)

    # Build new row
    new_row = {}
    # preserve column order in df_emb; fill pm1, pm2, proteins_id when available
    for col in df_emb.columns:
        if col == 'pm1':
            new_row[col] = centroid[0]
        elif col == 'pm2':
            new_row[col] = centroid[1]
        elif col == 'proteins_id':
            new_row[col] = new_id if new_id is not None else ''
        else:
            # other columns: fill with NaN
            new_row[col] = np.nan

    df_app = pd.concat([df_emb.reset_index(drop=True), pd.DataFrame([new_row])], ignore_index=True)

    out_path = out_embedding_path or embedding_path.replace('.csv', '.appended.csv')
    df_app.to_csv(out_path, index=False)
    logger.info("Wrote appended embedding to %s (shape=%s)", out_path, df_app.shape)

    return df_app, neighbors


def append_annotation_for_new_point(annotation_path, new_id=None, out_annotation_path=None,
                                    str_fill="New_point", num_fill=0):
    """Append a new annotation row to `annotation_path`.

    The new row will have `proteins_id` == new_id (or auto-generated) and all
    other columns filled with `str_fill` for non-numeric columns or `num_fill`
    for numeric columns.

    Returns the appended DataFrame.
    """
    if not os.path.exists(annotation_path):
        raise FileNotFoundError(f"Annotation file not found: {annotation_path}")

    df = pd.read_csv(annotation_path)

    # Ensure proteins_id column exists
    if 'proteins_id' not in df.columns:
        # insert as first column with sequential ids
        df.insert(0, 'proteins_id', range(len(df)))

    # Determine new_id if not provided
    if new_id is None:
        # try numeric max + 1 when possible
        try:
            # coerce to numeric where possible
            numeric_ids = pd.to_numeric(df['proteins_id'], errors='coerce')
            if numeric_ids.notna().all():
                new_id = int(numeric_ids.max()) + 1
            else:
                new_id = str(len(df))
        except Exception:
            new_id = str(len(df))

    # Build new row matching columns
    new_row = {}
    for col in df.columns:
        if col == 'proteins_id':
            new_row[col] = new_id
            continue

        # If column is numeric dtype (or can be coerced), fill with num_fill
        if pd.api.types.is_numeric_dtype(df[col]):
            new_row[col] = num_fill
        else:
            # For object / string columns fill with str_fill
            new_row[col] = str_fill

    df_app = pd.concat([df.reset_index(drop=True), pd.DataFrame([new_row])], ignore_index=True)

    out_path = out_annotation_path or annotation_path.replace('.csv', '.appended.csv')
    df_app.to_csv(out_path, index=False)
    logger.info("Wrote appended annotation to %s (shape=%s)", out_path, df_app.shape)

    return df_app

# Construct a padded numpy matrices for a given PSSM matrix
# This will be the function to change for embeddings
def construct_tensor(fpath):
    """Load a flattened numeric vector from a plain-text file.

    The original code used `.aamtx` files containing PSSM-like values;
    here we simply load and flatten the numeric array.
    """
    ansarr = np.loadtxt(fpath).reshape(-1)
    return np.array(ansarr)


def construct_tensor_from_embedding(fpath, option="mean"):
    """Load an embedding from a .pt (torch) file and reduce it to a vector.

    Supported keys in the saved file: `embedding`, `aae_embedding`.
    If the embedding has sequence length (L, D) it is mean-pooled.
    """
    # weights_only=False is required for .pt files that contain numpy arrays
    # (created before PyTorch 2.6 tightened the default).  These files come
    # from trusted local sources (embeddings/ directory).
    data = torch.load(fpath, weights_only=False)

    if "embedding" in data:
        emb = data["embedding"]
        if not torch.is_tensor(emb):
            emb = torch.tensor(emb)
        if emb.ndim == 2:
            emb = emb.mean(dim=0)
        elif emb.ndim != 1:
            raise ValueError(f"Unexpected shape {emb.shape} in {fpath}")
    elif "aae_embedding" in data:
        emb = data["aae_embedding"]
        if not torch.is_tensor(emb):
            emb = torch.tensor(emb)
        emb = emb.reshape(-1)
    else:
        raise KeyError(f"No recognized embedding key in {fpath}")

    return emb



def prepare_data(fpath, withroot = True, fmt='.aamtx'):
    # print([x[0] for x in os.walk(fpath)])
    # subfolders = [f.path for f in os.listdir(fpath) if f.is_dir() ]   
    # fmt = '.aamtx'
    proteins = [s for s in os.listdir(fpath) if fmt in s]
    n_proteins = len(proteins)
    logger = logging.getLogger(__name__)
    logger.info("%d proteins found in folder %s.", n_proteins - 1, fpath)

    if not withroot:
        proteins.remove(f"0{fmt}")
        n_proteins = len(proteins)
        logger.info("No root detected")

    protein_file = proteins[0]
    logger.debug("Example files: %s", proteins[:20])
    logger.debug("First file: %s", protein_file)
    fin = f'{fpath}/{protein_file}'    

    a = construct_tensor(fin)

    features = np.zeros([n_proteins, len(a)])
    labels = []
    logger.info("Prepare data: tensor construction")
    for i, protein_name in enumerate(proteins):
        #print(i, protein_name)
        fin = f'{fpath}/{protein_name}'
        features[i, :] = construct_tensor(fin)
        labels.append(protein_name.split('.')[0])
    logger.info("Prepare data: successfully terminated")
    return torch.Tensor(features), np.array(labels)


def prepare_embedding_data(fpath, withroot = True, fmt='.pt'):
    '''Same function as prepare_data, but adapted to handle embeddings

    Args
    ---
    fpath : str
        Path to folder with the embeddings
    withroot : boolean
        Protein family root ?
    fmt : str
        What is the file type of the embeddings ?

    Return
    ---
    torch.Tensor(features) : matrix of fixed embeddings
    np.array(labels) : corresponding labels
    '''
    proteins = [s for s in os.listdir(fpath) if fmt in s]
    n_proteins = len(proteins)
    logger = logging.getLogger(__name__)
    logger.info("%d proteins found in folder %s.", n_proteins, fpath)
    logger.debug("files: %s", proteins)

    # if not withroot:
    #     proteins.remove("0.txt")
    #     n_proteins = len(proteins)
    #     print("No root detected")

    protein_file = proteins[0]
    logger.debug("First file: %s", protein_file)
    fin = f'{fpath}/{protein_file}'    

    a = construct_tensor_from_embedding(fin)

    features = np.zeros([n_proteins, len(a)])
    labels = []
    logger.info("Prepare data: tensor construction")
    for i, protein_name in enumerate(proteins):
        fin = f'{fpath}/{protein_name}'
        features[i, :] = construct_tensor_from_embedding(fin)
        labels.append(protein_name.split('.')[0])
    logger.info("Prepare data: successfully terminated")
    return torch.Tensor(features), np.array(labels)


# def prepare_data(fin, with_labels=True, normalize=False, n_pca=0):
#     """
#     Reads a dataset in CSV format from the ones in datasets/
#     """
#     df = pd.read_csv(fin + '.csv', sep=',')
#     n = len(df.columns)

#     if with_labels:
#         x = np.double(df.values[:, 0:n - 1])
#         labels = df.values[:, (n - 1)]
#         labels = labels.astype(str)
#         colnames = df.columns[0:n - 1]
#     else:
#         x = np.double(df.values)
#         labels = ['unknown'] * np.size(x, 0)
#         colnames = df.columns

#     n = len(colnames)

#     idx = np.where(np.std(x, axis=0) != 0)[0]
#     x = x[:, idx]

#     if normalize:
#         s = np.std(x, axis=0)
#         s[s == 0] = 1
#         x = (x - np.mean(x, axis=0)) / s

#     if n_pca:
#         if n_pca == 1:
#             n_pca = n

#         nc = min(n_pca, n)
#         pca = PCA(n_components=nc)
#         x = pca.fit_transform(x)

#     labels = np.array([str(s) for s in labels])

#     return torch.DoubleTensor(x), labels


def connect_knn(KNN, distances, n_components, labels):
    """Connect KNN components by adding minimal inter-component edges.

    This is a utility used when a connected KNN graph is required for
    downstream Laplacian/RFA computation.
    """
    cur_comp = 0
    while n_components > 1:
        idx_cur = np.where(labels == cur_comp)[0]
        idx_rest = np.where(labels != cur_comp)[0]
        d = distances[np.ix_(idx_cur, idx_rest)]
        ia, ja = np.where(d == np.min(d))
        i = ia[0]
        j = ja[0]

        KNN[idx_cur[i], idx_rest[j]] = distances[idx_cur[i], idx_rest[j]]
        KNN[idx_rest[j], idx_cur[i]] = distances[idx_rest[j], idx_cur[i]]

        nearest_comp = labels[idx_rest[j]]
        labels[labels == nearest_comp] = cur_comp
        n_components -= 1

    return KNN


def compute_rfa(features,  distance_matrix=None, mode='features', k_neighbours=15, distfn='sym',
                connected=False, sigma=1.0, distlocal='minkowski'):
    """
    Computes the target RFA similarity matrix. The RFA matrix of
    similarities relates to the commute time between pairs of nodes, and it is
    built on top of the Laplacian of a single connected component k-nearest
    neighbour graph of the data.
    """
    start = timeit.default_timer()
    if mode == 'features':
        KNN = kneighbors_graph(features,
                               k_neighbours,
                               mode='distance',
                               metric=distlocal,
                               include_self=False,
                               ).toarray()

        if 'sym' in distfn.lower():
            KNN = np.maximum(KNN, KNN.T)
        else:
            KNN = np.minimum(KNN, KNN.T)    

        n_components, labels = csgraph.connected_components(KNN)

        if connected and (n_components > 1):
            distances = pairwise_distances(features, metric=distlocal)
            KNN = connect_knn(KNN, distances, n_components, labels)
    else:
        KNN = features    

    if distlocal == 'minkowski':
        # When features are available, normalize by feature dimensionality as before.
        # If only a precomputed distance matrix was provided (features is None),
        # fall back to a simple normalization to avoid attribute errors.
        if features is not None or distance_matrix is not None:
            # features might be a torch Tensor or numpy array
            try:
                n_feat = features.shape[1] if features is not None else distance_matrix.shape[1]
            except Exception:
                # fallback if shape is not available
                n_feat = 1
            denom = sigma * n_feat
        else:
            denom = sigma

        S = np.exp(-KNN / denom)
    else:
        S = np.exp(-KNN / sigma)

    S[KNN == 0] = 0
    logger = logging.getLogger(__name__)
    logger.info("Computing laplacian...")
    L = csgraph.laplacian(S, normed=False)
    logger.info("Laplacian computed in %.2f sec", (timeit.default_timer() - start))

    logger.info("Computing RFA...")
    start = timeit.default_timer()
    RFA = np.linalg.inv(L + np.eye(L.shape[0]))
    # Replace NaNs (if any) with zeros
    RFA[np.isnan(RFA)] = 0.0

    logger.info("RFA computed in %.2f sec", (timeit.default_timer() - start))

    return torch.Tensor(RFA)


def compute_rfa_w_custom_distance(features=None, distance_matrix=None,
                                  k_neighbours=15, distfn='sym', connected=False,
                                  sigma=1.0, distlocal='minkowski', output_path=None):
                                  
    """
    Computes the target RFA similarity matrix. The RFA matrix of
    similarities relates to the commute time between pairs of nodes, and it is
    built on top of the Laplacian of a single connected component k-nearest
    neighbour graph of the data.
    """
    start = timeit.default_timer()

    # # Verify that kneighbors_graph can also take a distance matrix as input
    # # and that both KNN matrices are equal:
    # if features is not None:
    #     # Calculate a distance matrix with pairwise_distances from sklearn
    #     sklearn_distance_matrix = pairwise_distances(features, metric=distlocal)
    #     # Compute the KNN matrices
    #     KNN_distance_matrix = kneighbors_graph(sklearn_distance_matrix,
    #                                            k_neighbours,
    #                                            mode='distance',
    #                                            metric='precomputed',
    #                                            include_self=False).toarray()
    #     KNN_features = kneighbors_graph(features,
    #                                     k_neighbours,
    #                                     mode='distance',
    #                                     metric=distlocal,
    #                                     include_self=False).toarray()
    #     # Verify that both KNN matrices are equal
    #     same_graph = np.array_equal(KNN_distance_matrix, KNN_features)
    #     print(f"KNN matrices are equal: {same_graph}")  # True
    #     KNN = KNN_distance_matrix if same_graph else KNN_features
    # Indeed, kneighbors_graph can take a distance matrix as input if metric='precomputed'
    # The valid distance metrics for kneighbors_graph and pairwise distances are:
    # ['cityblock', 'cosine', 'euclidean', 'haversine', 'l1', 'l2', 'manhattan', 'nan_euclidean', 'precomputed']
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.distance_metrics.html#sklearn.metrics.pairwise.distance_metrics
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances.html

    # Compute the KNN matrix using either features or a provided distance matrix
    if features is not None or distance_matrix is not None:
        # Use distance_matrix if provided, otherwise use the features
        data = distance_matrix if distance_matrix is not None else features
        metric = 'precomputed' if distance_matrix is not None else distlocal
        KNN = kneighbors_graph(data,
                               k_neighbours,
                               mode='distance',
                               metric=metric,
                               include_self=False,
                               ).toarray()
        # Symmetrize the KNN matrix
        if 'sym' in distfn.lower():
            KNN = np.maximum(KNN, KNN.T)
        else:
            KNN = np.minimum(KNN, KNN.T)
        # Handle connected components
        n_components, labels = csgraph.connected_components(KNN)
        if connected and (n_components > 1):
            # Use the features to calculate pairwise distances if needed
            distances = pairwise_distances(features, metric=distlocal) if distance_matrix is None else data
            KNN = connect_knn(KNN, distances, n_components, labels)
            # Save distance matrix as CSV file, Numpy array
            distances_path = os.path.join(output_path, 'distance_matrix.csv')
            np.savetxt(distances_path, distances, delimiter=",")
        # Save the KNN matrix as CSV file, NumPy array
        if output_path is not None:
            KNN_output_path = os.path.join(output_path, 'KNN_matrix.csv')
            np.savetxt(KNN_output_path, KNN, delimiter=",")
            logger.info("KNN matrix CSV file saved to %s", KNN_output_path)

    # # If mode is not 'features' and no distance_matrix is provided, assume KNN is already computed
    # else:
    #     KNN = features

    # Compute the similarity matrix S
    if distlocal == 'minkowski':
        # When features are available, normalize by feature dimensionality as before.
        # If only a precomputed distance matrix was provided (features is None),
        # fall back to a simple normalization to avoid attribute errors.
        if features is not None or distance_matrix is not None:
            # features might be a torch Tensor or numpy array
            try:
                n_feat = features.shape[1] if features is not None else distance_matrix.shape[1]
            except Exception:
                # fallback if shape is not available
                n_feat = 1
            denom = sigma * n_feat
        else:
            denom = sigma

        S = np.exp(-KNN / denom)
    else:
        S = np.exp(-KNN / sigma)

    # Compute the Laplacian
    S[KNN == 0] = 0
    logger.info("Computing laplacian...")
    L = csgraph.laplacian(S, normed=False)
    logger.info("Laplacian computed in %.2f sec", (timeit.default_timer() - start))

    # Compute the RFA matrix
    logger.info("Computing RFA...")
    start = timeit.default_timer()
    RFA = np.linalg.inv(L + np.eye(L.shape[0]))
    # Replace NaNs (if any) with zeros
    RFA[np.isnan(RFA)] = 0.0
    logger.info("RFA computed in %.2f sec", (timeit.default_timer() - start))

    return torch.Tensor(RFA)


