# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""CLI entrypoint for building Poincaré maps.

This module orchestrates data preparation (features or precomputed distance
matrices), RFA computation and training of the Poincaré embedding.

The script intentionally keeps I/O and orchestration logic here while
computation routines live in `data.py` and optimization/Model code in
`model.py` / `train.py`.
"""

#########################################################################################
############################  LIBRAIRIES & IMPORT  ######################################
#########################################################################################

import argparse
import logging
import os
import timeit
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset

from data import (
    prepare_data,
    prepare_embedding_data,
    compute_rfa,
    compute_rfa_w_custom_distance,
)
from model import PoincareEmbedding, PoincareDistance, poincare_translation
from rsgd import RiemannianSGD
from train import train
from poincare_maps import plotPoincareDisc



#########################################################################################
######################################  SETUP  ##########################################
#########################################################################################

# Minimal logging setup used instead of prints to make messages easier to
# filter/redirect in downstream tools.
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


#########################################################################################
###############################  USEFULL FUNCTIONS  #####################################
#########################################################################################


### This function was added by Tatina Galochkina in order to improve reproducibility of the restuls 
def set_seed(seed: int = 42) -> None:
    """Set deterministic seeds for NumPy, PyTorch and environment.

    This improves reproducibility across runs. Note that full reproducibility
    across platforms/hardware is not guaranteed, but these settings reduce
    nondeterminism from common sources.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info("Random seed set as %d", seed)


def create_output_name(opt) -> Tuple[str, str]:
    """Compose a human-readable title and a filesystem-safe filename base.

    Returns (title, filename_base).
    """
    titlename = (
        f"dist={opt.distfn}, metric={opt.distlocal}, knn={opt.knn}, "
        f"loss={opt.lossfn} sigma={opt.sigma:.2f}, gamma={opt.gamma:.2f}, n_pca={opt.pca}"
    )

    os.makedirs(opt.output_path, exist_ok=True)

    filename = os.path.join(
        opt.output_path,
        f"PM{opt.knn:d}"
        f"sigma={opt.sigma:.2f}"
        f"gamma={opt.gamma:.2f}"
        f"{opt.distlocal}pca={opt.pca:d}_seed{opt.seed}",
    )

    return titlename, filename

#def get_tree_colors(opt, labels, tree_cl_name):
#    pkl_file = open(f'{tree_cl_name}.pkl', 'rb')
#    colors = pickle.load(pkl_file)
#    colors_keys = [str(k) for k in colors.keys()]
#    colors_val = [str(k) for k in colors.values()]
#    colors = dict(zip(colors_keys, colors_val))
#    pkl_file.close()
#    tree_levels = []
#    for l in labels:
#        if l == 'root':
#            tree_levels.append('root')
#        else:
#            tree_levels.append(colors[l])
#
#    tree_levels = np.array(tree_levels)
#    n_tree_levels = len(np.unique(tree_levels))
#    current_palette = sns.color_palette("husl", n_tree_levels)
#    color_dict = dict(zip(np.unique(tree_levels), current_palette))
#    sns.palplot(current_palette)
#    color_dict[-1] = '#bdbdbd'
#    color_dict['root'] = '#000000'
#    return tree_levels, color_dict


def parse_args():
    parser = argparse.ArgumentParser(
        description='Adaptation of Poincare maps for MSA')
    
    parser.add_argument('--method', help='Method to choose : pssm, plm, plm_aae, RFA_matrix, distance_matrix', type=str, default="pssm")

    parser.add_argument('--dim', help='Embedding dimension', type=int, default=2)

    # Path to input files
    parser.add_argument('--input_path', help='Path to dataset to embed', type=str, 
        default='examples/thioredoxins/fasta0.9/')

    # Path to output folder for figures
    parser.add_argument('--output_path', help='Path to dataset to embed', type=str, 
        default='results/')
    
    # Use mfasta or plm embeddings ?
    parser.add_argument('--plm_embedding', help='Type of input data that should be used', type=str, 
        default='False')
    
    # Path to output folder for intermediate matrices
    parser.add_argument('--matrices_output_path', help='Path to save KNN and RFA matrices', type=str)

    # Plot results ?
    parser.add_argument('--plot',
        help='Flag True or False, if you want to plot the output.', type=str, 
        default='True')
    
    # Checkout frequency
    parser.add_argument('--checkout_freq',
        help='Checkout frequency (in epochs) to show intermediate results', 
        type=int, default=10)


    parser.add_argument('--tree', 
        help='File with phylogenetic trees', type=str, default=5)
    parser.add_argument('--function', 
        help='Protein by function', type=str, default='glob-name')
    parser.add_argument('--seed',
        help='Random seed', type=int, default=0)
    parser.add_argument('--labels', help='array containing the feature labels in the same\
        order as in the features dataset used to compute the provided distance matrix',
        type=str)
    parser.add_argument('--distance_matrix',
        help='Path to the CSV file containing the precomputed distance matrix',
        type=str, default=None)
    parser.add_argument('--normalize',
        help='Apply z-transform to the data', type=int, default=0)
    parser.add_argument('--pca',
        help='Apply pca for data preprocessing (if pca=0, no pca)', 
        type=int, default=0)
    parser.add_argument('--distlocal', 
        help='Distance function (minkowski, cosine)', 
        type=str, default='cosine')
    parser.add_argument('--distfn', 
        help='Distance function (Euclidean, MFImixSym, MFI, MFIsym)', 
        type=str, default='MFIsym')
    parser.add_argument('--distr', 
        help='Target distribution (laplace, gaussian, student)', 
        type=str, default='laplace')
    parser.add_argument('--lossfn', help='Loss funstion (kl, klSym)',
        type=str, default='klSym')
    parser.add_argument('--iroot',
        help='Index of the root cell', type=int, default=0)
    parser.add_argument('--rotate',
        help='use 0 element for calculations or not', action='store_true')
    parser.add_argument('--knn', 
        help='Number of nearest neighbours in KNN', type=int, default=5)
    parser.add_argument('--connected',
        help='Force the knn graph to be connected', type=int, default=1)
    parser.add_argument('--sigma',
        help='Bandwidth in high dimensional space', type=float, default=1.0)
    parser.add_argument('--gamma',
        help='Bandwidth in low dimensional space', type=float, default=2.0)

    # optimization parameters
    parser.add_argument('--lr',
        help='Learning rate', type=float, default=0.1)
    parser.add_argument('--lrm',
        help='Learning rate multiplier', type=float, default=1.0)
    parser.add_argument('--epochs',
        help='Number of epochs', type=int, default=1000)
    parser.add_argument('--batchsize',
        help='Batchsize', type=int, default=4)
    parser.add_argument('--burnin',
        help='Duration of burnin', type=int, default=500)

    parser.add_argument('--earlystop',
        help='Early stop  of training by epsilon. If 0, continue to max epochs', 
        type=float, default=0.0001)

    parser.add_argument('--debugplot',
        help='Plot intermidiate embeddings every N iterations',
        type=int, default=200)
    parser.add_argument('--logfile',
        help='Use GPU', type=str, default='Logs')
    args = parser.parse_args()

    args.plot = bool(args.plot)
    return args

def poincare_map(opt):
    # read and preprocess the dataset
# Configure device and random seed
    opt.cuda = torch.cuda.is_available()
    logger.info("CUDA available: %s", opt.cuda)
    set_seed(opt.seed)
#    torch.manual_seed(opt.seed)

    features, labels = prepare_data(opt.input_path, withroot=opt.rotate)
    # if not (opt.tree is None):
    #     tree_levels, color_dict = get_tree_colors(
    #         opt, labels, 
    #         f'{opt.input_path}/{opt.family}_tree_cluster_{opt.tree}')
    # else:
    #     color_dict = None
    #     tree_levels = None

    # compute matrix of RFA similarities
    try:
        # First try a pandas read assuming there may be an index column.
        # However pandas by default treats the first row as header which can
        # incorrectly drop one data row if the file has no header. To detect
        # that case we compare the dataframe rowcount to the raw file linecount
        # and fall back to a header-less read if they disagree.
        df = pd.read_csv(opt.distance_matrix, index_col=0)
        # count raw lines in file to detect header misinterpretation
        try:
            with open(opt.distance_matrix, 'r') as fh:
                raw_lines = sum(1 for _ in fh)
        except Exception:
            raw_lines = None

        # If pandas read appears to have consumed a header (raw_lines == df.shape[0] + 1)
        # then re-read without header to preserve all numeric rows.
        if raw_lines is not None and raw_lines == df.shape[0] + 1:
            df = pd.read_csv(opt.distance_matrix, header=None)

        # If the CSV appears square, use it; otherwise try numeric loadtxt fallback
        if df.shape[0] == df.shape[1]:
            distance_matrix = df.values
            if opt.labels is None:
                # If the dataframe had an index, use it; otherwise generate string indices
                try:
                    labels = df.index.astype(str).to_numpy()
                except Exception:
                    labels = np.array([str(i) for i in range(distance_matrix.shape[0])])
                # Save labels to matrices_output_path so downstream code can find them
                if opt.matrices_output_path is not None:
                    os.makedirs(opt.matrices_output_path, exist_ok=True)
                    labels_path = os.path.join(opt.matrices_output_path, "labels.csv")
                    np.savetxt(labels_path, labels, delimiter=",", fmt="%s")
                    logger.info("labels CSV file saved to %s", labels_path)
        else:
            # Not a square dataframe — fallback to numpy load
            distance_matrix = np.loadtxt(opt.distance_matrix, delimiter=',')
    except Exception:
        # Fallback: plain numeric CSV
        distance_matrix = np.loadtxt(opt.distance_matrix, delimiter=',')

    RFA = compute_rfa(
        features,
        distance_matrix,
        mode=opt.mode,
        k_neighbours=opt.knn,
        distfn=opt.distfn,
        distlocal= opt.distlocal,
        connected=opt.connected,
        sigma=opt.sigma
        )
    logger.debug("RFA tensor shape: %s", getattr(RFA, 'shape', None))
    if opt.batchsize < 0:
        opt.batchsize = min(512, int(len(RFA) / 10))
        logger.info("batchsize set to %d", opt.batchsize)

    opt.lr = opt.batchsize / 16 * opt.lr

    titlename, fout = create_output_name(opt)

    indices = torch.arange(len(RFA))
    if opt.cuda:
        indices = indices.cuda()
        RFA = RFA.cuda()

    dataset = TensorDataset(indices, RFA)

    # instantiate our Embedding predictor
    predictor = PoincareEmbedding(
        len(dataset),
        opt.dim,
        dist=PoincareDistance,
        max_norm=1,
        Qdist=opt.distr, 
        lossfn = opt.lossfn,
        gamma=opt.gamma,
        cuda=opt.cuda
        )

    # instantiate the Riemannian optimizer 
    t_start = timeit.default_timer()
    optimizer = RiemannianSGD(predictor.parameters(), lr=opt.lr)

    # train predictor
    logger.info("Starting training...")
    embeddings, loss, epoch = train(
        predictor, dataset, optimizer, opt, fout=fout, earlystop=opt.earlystop
    )

    df_pm = pd.DataFrame(embeddings, columns=["pm1", "pm2"])
    df_pm["proteins_id"] = labels

# Optionally recenter the disk at the provided root index
    if opt.rotate:
        idx_root = np.where(df_pm["proteins_id"] == str(opt.iroot))[0][0]
        logger.info("Recentering poincare disk at %s", opt.iroot)
#        print("root index: ", idx_root)
        poincare_coord_rot = poincare_translation(
            -embeddings[idx_root, :], embeddings)
        df_rot = df_pm.copy()
        df_rot['pm1'] = poincare_coord_rot[:, 0]
        df_rot['pm2'] = poincare_coord_rot[:, 1]
        df_rot.to_csv(fout + ".csv", sep=",", index=False)
    else:
        df_pm.to_csv(fout + ".csv", sep=",", index=False)

    t = timeit.default_timer() - t_start
    titlename = f"\nloss = {loss:.3e}\ntime = {t/60:.3f} min"
    logger.info(titlename)

    plotPoincareDisc(
        embeddings, title_name=titlename, file_name=fout, d1=5.5, d2=5.0, bbox=(1.2, 1.0), leg=False
    )

    # idx_root = np.where(tree_levels == 'root')[0]
    # poincare_coord_rot = poincare_translation(-embeddings[idx_root, :], embeddings)


    # if not (opt.function is None):
    #     for f in ['glob_tree_cluster_1']:
    #         fun_levels, color_dict_fun = get_tree_colors(opt, labels, f'{opt.input_path}/{opt.family}/{f}')
    #         plotPoincareDisc(poincare_coord_rot, fun_levels, 
    #                            title_name=titlename,
    #                            labels=fun_levels, 
    #                            coldict=color_dict_fun, 
    #                            file_name=f'{fout}_rotate_{f}', 
    #                            d1=8.5, d2=8.0, bbox=(1., 1.), leg=False)

    # else:
    #     color_dict_fun = None
    #     fun_levels = None
    
    # for t in range(1, 6):
    #     tree_levels, color_dict = get_tree_colors(opt, labels, f'{opt.input_path}/{opt.family}/{opt.family}_tree_cluster_{t}')
    #     if len(np.unique(tree_levels)) < 25:
    #         leg = True
    #     else:
    #         leg = False
    #     plotPoincareDisc(poincare_coord_rot, labels, 
    #                        title_name=titlename,
    #                        labels=tree_levels, 
    #                        coldict=color_dict, file_name=f'{fout}_rotate_cut{t}', d1=8.5, d2=8.0, bbox=(1.2, 1.), leg=leg)




#########################################################################################
##################################### MAIN FUNCTION #####################################
#########################################################################################

def poincare_map_w_custom_distance(opt):
    # read and preprocess the dataset
# Configure device and seed
    opt.cuda = torch.cuda.is_available()
    logger.info("CUDA available: %s", opt.cuda)
    set_seed(opt.seed)
    torch.manual_seed(opt.seed)

##########################
## I - Data preparation ##
##########################

    if opt.method != "RFA_matrix" :

#------------------------------------
# I - a) With an mfasta file (pssm) -
#------------------------------------
        # Features assignment only if a precomputed distance matrix is not provided
        if opt.distance_matrix is None and opt.plm_embedding == 'False' :
            features, labels = prepare_data(opt.input_path, withroot=opt.rotate)
            logger.debug("labels: %s", labels)

            # Download features as CSV file, Numpy array
            features_path = os.path.join(opt.matrices_output_path, "features.csv")
            np.savetxt(features_path, features, delimiter=",")
            logger.info("features CSV file saved to %s", features_path)

            # Download labels as CSV file, Numpy array
            labels_path = os.path.join(opt.matrices_output_path, "labels.csv")
            np.savetxt(labels_path, labels, delimiter=",", fmt="%s")
            logger.info("labels CSV file saved to %s", labels_path)

            distance_matrix = None

#--------------------------------------
# I - b) With an embedding file (plm) -
#--------------------------------------
        elif opt.distance_matrix is None and opt.plm_embedding == 'True':
            features, labels = prepare_embedding_data(opt.input_path, withroot=opt.rotate)
            logger.debug("labels: %s", labels)

            # Download features as CSV file, Numpy array
            features_path = os.path.join(opt.matrices_output_path, "features.csv")
            np.savetxt(features_path, features, delimiter=",")
            logger.info("features CSV file saved to %s", features_path)

            # Download labels as CSV file, Numpy array
            labels_path = os.path.join(opt.matrices_output_path, "labels.csv")
            np.savetxt(labels_path, labels, delimiter=",", fmt="%s")
            logger.info("labels CSV file saved to %s", labels_path)

            distance_matrix = None

#--------------------------------
# I - c) With a distance matrix -
#--------------------------------
        else:
            # If a user provides a precomputed distance matrix we try to read it robustly.
            # Accepts CSVs with an index/column labels (square dataframes) or plain numeric CSVs.
            features = None
            distance_matrix = None
            labels = None

            if opt.distance_matrix is None:
                raise AttributeError('distance_matrix path expected when entering this branch')

            # Try to read with pandas to preserve labels if present
            try:
                # First try a pandas read assuming there may be an index column.
                # However pandas by default treats the first row as header which can
                # incorrectly drop one data row if the file has no header. To detect
                # that case we compare the dataframe rowcount to the raw file linecount
                # and fall back to a header-less read if they disagree.
                df = pd.read_csv(opt.distance_matrix, index_col=0)
                # count raw lines in file to detect header misinterpretation
                try:
                    with open(opt.distance_matrix, 'r') as fh:
                        raw_lines = sum(1 for _ in fh)
                except Exception:
                    raw_lines = None

                # If pandas read appears to have consumed a header (raw_lines == df.shape[0] + 1)
                # then re-read without header to preserve all numeric rows.
                if raw_lines is not None and raw_lines == df.shape[0] + 1:
                    df = pd.read_csv(opt.distance_matrix, header=None)

                # If the CSV appears square, use it; otherwise try numeric loadtxt fallback
                if df.shape[0] == df.shape[1]:
                    distance_matrix = df.values
                    if opt.labels is None:
                        # If the dataframe had an index, use it; otherwise generate string indices
                        try:
                            labels = df.index.astype(str).to_numpy()
                        except Exception:
                            labels = np.array([str(i) for i in range(distance_matrix.shape[0])])
                        # Save labels to matrices_output_path so downstream code can find them
                        if opt.matrices_output_path is not None:
                            os.makedirs(opt.matrices_output_path, exist_ok=True)
                            labels_path = os.path.join(opt.matrices_output_path, "labels.csv")
                            np.savetxt(labels_path, labels, delimiter=",", fmt="%s")
                            logger.info("labels CSV file saved to %s", labels_path)
                else:
                    # Not a square dataframe — fallback to numpy load
                    distance_matrix = np.loadtxt(opt.distance_matrix, delimiter=',')
            except Exception:
                # Fallback: plain numeric CSV
                distance_matrix = np.loadtxt(opt.distance_matrix, delimiter=',')

            # If labels were not inferred from the CSV, try to load from opt.labels
            if labels is None:
                if opt.labels is not None:
                    labels = np.loadtxt(opt.labels, delimiter=',', dtype=str)
                else:
                    # No labels provided: create default numeric string labels 0..n-1
                    labels = np.array([str(i) for i in range(distance_matrix.shape[0])])

            # Ensure output directories exist
            if opt.matrices_output_path is not None:
                os.makedirs(opt.matrices_output_path, exist_ok=True)
            if not os.path.exists(opt.output_path):
                os.makedirs(opt.output_path)
            
                    # Diagnostic: print shapes and basic checks to help debug label/matrix mismatches
            try:
                dm_shape = np.array(distance_matrix).shape
            except Exception:
                dm_shape = None
            logger.debug("distance_matrix shape = %s", dm_shape)
            logger.debug("inferred labels length = %s", len(labels) if labels is not None else "None")
            # Validate distance matrix vs labels
            if dm_shape is not None:
                if len(dm_shape) != 2 or dm_shape[0] != dm_shape[1]:
                    raise ValueError(f"Provided distance_matrix must be a square 2D array. Got shape {dm_shape}.")
                if labels is not None and len(labels) != dm_shape[0]:
                    raise ValueError(f"Labels length ({len(labels)}) does not match distance_matrix size ({dm_shape[0]}). Please provide matching labels or a correctly sized distance matrix.")

    # if not (opt.tree is None):
    #     tree_levels, color_dict = get_tree_colors(
    #         opt, labels,
    #         f'{opt.input_path}/{opt.family}_tree_cluster_{opt.tree}')
    # else:
    #     color_dict = None
    #     tree_levels = None


#########################################################################
## II - Creation of KNN Graph & Computation of RFA similarities matrix ##
#########################################################################

#---------------------------------------------
# II - a) Using pssm, plm or distance matrix -
#---------------------------------------------
    if opt.method != "RFA_matrix" :
        RFA = compute_rfa_w_custom_distance(
            features,
            distance_matrix,
            k_neighbours=opt.knn,
            distfn=opt.distfn,
            distlocal=opt.distlocal,
            connected=opt.connected,
            sigma=opt.sigma,
            output_path=opt.matrices_output_path,
        )
        logger.info("RFA matrix computed (tensor shape %s)", tuple(RFA.shape))

        # Download RFA matrix as CSV file, NumPy array
        RFA_matrix_path = os.path.join(opt.matrices_output_path, "RFA_matrix.csv")
        np.savetxt(RFA_matrix_path, RFA, delimiter=",")
        logger.info("RFA matrix CSV file saved to %s", RFA_matrix_path)

#---------------------------
# II - b) Using RFA matrix -
#---------------------------
    else:
        # If the user asked to use a precomputed RFA matrix, load it from disk
        RFA_matrix_path = os.path.join(opt.matrices_output_path, "RFA_matrix.csv")
        RFA_txt = np.loadtxt(RFA_matrix_path, delimiter=",")
        RFA = torch.tensor(RFA_txt, dtype=torch.float32)
        logger.info("RFA matrix CSV file loaded from %s", RFA_matrix_path)

        # Download labels as CSV file, Numpy array
        labels_path = os.path.join(opt.matrices_output_path, "labels.csv")
        labels = np.loadtxt(labels_path, delimiter=",", dtype=str)
        logger.info("labels CSV file loaded from %s", labels_path)

#---------------------------------------
# II - c) Tensorization of RFA  matrix -
#---------------------------------------
    # Continue using RFA as a tensor in the rest of the code
    if opt.batchsize < 0:
        opt.batchsize = min(512, int(len(RFA) / 10))
        logger.info("batchsize = %d", opt.batchsize)

    opt.lr = opt.batchsize / 16 * opt.lr

    titlename, fout = create_output_name(opt)

    indices = torch.arange(len(RFA))
    if opt.cuda:
        indices = indices.cuda()
        RFA = RFA.cuda()

    dataset = TensorDataset(indices, RFA)


################################################## 
## III - Instanciation of predictor & optimizer ##
##################################################

#----------------------------------------------------
# III - a) Instantiation of the embedding predictor -
#----------------------------------------------------
    predictor = PoincareEmbedding(
        len(dataset),
        opt.dim,
        dist=PoincareDistance,
        max_norm=1,
        Qdist=opt.distr, 
        lossfn = opt.lossfn,
        gamma=opt.gamma,
        cuda=opt.cuda
        )

#------------------------------------------
# III - b) Instantiation of the optimizer -
#------------------------------------------
    t_start = timeit.default_timer()
    optimizer = RiemannianSGD(predictor.parameters(), lr=opt.lr)


####################################
## IV - Training of the predictor ##
####################################

    logger.info('Starting training...')
    embeddings, loss, epoch = train(
        predictor,
        dataset,
        optimizer,
        opt,
        fout=fout,
        earlystop=opt.earlystop
        )

    df_pm = pd.DataFrame(embeddings, columns=["pm1", "pm2"])


#############################################
## V - Loading & preparation of the labels ##
#############################################

    # Prefer the labels variable populated earlier (when distance matrix provided we try to infer/save labels).
    if 'labels' in locals() and labels is not None:
        labels_arr = np.array(labels, dtype=str)
    else:
        if opt.labels is not None and os.path.exists(opt.labels):
            try:
                labels_arr = np.loadtxt(opt.labels, delimiter=',', dtype=str)
            except Exception:
                try:
                    labels_arr = pd.read_csv(opt.labels, header=None).iloc[:, 0].astype(str).to_numpy()
                except Exception:
                    labels_arr = np.array([], dtype=str)
        else:
            labels_arr = np.array([], dtype=str)

    # Clean whitespace and drop empty labels
    labels_arr = np.array([str(x).strip() for x in labels_arr])
    labels_arr = labels_arr[labels_arr != '']

    n_emb = len(df_pm)

    # Diagnostic info to help find origin of mismatch
    logger.debug("embeddings length = %d", n_emb)
    try:
        if 'distance_matrix' in locals() and distance_matrix is not None:
            logger.debug("Debug: distance_matrix shape = %s", np.array(distance_matrix).shape)
    except Exception:
        pass
    try:
        if 'features' in locals() and features is not None:
            try:
                logger.debug("Debug: features shape = %s", features.shape)
            except Exception:
                pass
    except Exception:
        pass

    # Strict check: labels must match number of embeddings. Fail early with clear message to let user fix data.
    if len(labels_arr) != n_emb:
        details = []
        details.append(f"embeddings_len={n_emb}")
        details.append(f"labels_len={len(labels_arr)}")
        if 'distance_matrix' in locals() and distance_matrix is not None:
            details.append(f"distance_matrix_shape={np.array(distance_matrix).shape}")
        if 'features' in locals() and features is not None:
            try:
                details.append(f"features_shape={features.shape}")
            except Exception:
                pass
        detail_msg = "; ".join(details)
        raise ValueError(
            f"Mismatch between number of embeddings and labels. {detail_msg}. "
            "Please ensure your labels correspond to the rows/columns of your input distance/features."
        )

    df_pm['proteins_id'] = labels_arr
    # df_pm = df_pm.sort_values(by='proteins_id')

##############################################
## VI - Creation of the final Poincarre csv ##
##############################################

#-----------------------------------------------
# VI - a) Changing the root if one is provided -
#-----------------------------------------------
    if opt.rotate:
        idx_root = np.where(df_pm["proteins_id"] == str(opt.iroot))[0][0]
        logger.info("Recentering poincare disk at %s", opt.iroot)
        print("root index: ", idx_root)
        poincare_coord_rot = poincare_translation(
            -embeddings[idx_root, :], embeddings)
        df_rot = df_pm.copy()
        df_rot['pm1'] = poincare_coord_rot[:, 0]
        df_rot['pm2'] = poincare_coord_rot[:, 1]
        df_rot.to_csv(fout + '.csv', sep=',', index=False)

#---------------------------
# VI - b) No root provided -
#---------------------------
    else:
        df_pm.to_csv(fout + '.csv', sep=',', index=False)




    t = timeit.default_timer() - t_start
    titlename = f"\nloss = {loss:.3e}\ntime = {t/60:.3f} min"
    logger.info(titlename)

    plotPoincareDisc(
        embeddings, 
        title_name=titlename,
        file_name=fout, 
        d1=5.5, d2=5.0, 
        bbox=(1.2, 1.),
        leg=False
        )

if __name__ == "__main__":
    args = parse_args()
    poincare_map_w_custom_distance(args)
