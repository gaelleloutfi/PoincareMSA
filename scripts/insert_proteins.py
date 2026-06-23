#!/usr/bin/env python3
"""
Insert new proteins into an existing Poincaré map.

The hyperparameters --gamma, --knn, and --distlocal must match the values
used when the original map was trained.  These are NOT stored in the
embedding CSV, so you must supply them explicitly (or rely on the defaults
if the map was built with default settings).

Typical usage
-------------
  python scripts/insert_proteins.py \\
      --embedding_csv results/PM5sigma=1.00gamma=2.00cospca=0_seed0.csv \\
      --existing_features examples/globins/plm/ \\
      --new_features path/to/new_proteins_plm/ \\
      --gamma 2.0 --knn 5 --distlocal cosine \\
      --strategy independent \\
      --output_csv results/PM5sigma=1.00gamma=2.00cospca=0_seed0_inserted.csv

Feature inputs
--------------
Both --existing_features and --new_features accept:
  * A folder of PLM embedding files (*.pt, one file per protein).
    Protein name = filename stem (e.g. "1abc.pt" → protein "1abc").
  * A CSV file where:
      - If the last column is non-numeric it is treated as protein names.
      - Otherwise the first column is treated as protein names.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch  # noqa: F401 — ensure torch is importable before model imports

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_BUILD_PKG = _PROJECT_ROOT / "scripts" / "build_poincare_map"

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
logger = logging.getLogger("insert_proteins")


# ---------------------------------------------------------------------------
# Feature loading
# ---------------------------------------------------------------------------

def _load_csv_features(path: Path) -> tuple[np.ndarray, list[str]]:
    """Load features from a CSV file. Returns (feats, labels)."""
    df = pd.read_csv(path)
    # Detect label column
    last_col = df.iloc[:, -1]
    if last_col.dtype == object or not np.issubdtype(last_col.dtype, np.number):
        labels = last_col.astype(str).tolist()
        feats = df.iloc[:, :-1].values.astype(np.float64)
    else:
        first_col = df.iloc[:, 0]
        if first_col.dtype == object or not np.issubdtype(first_col.dtype, np.number):
            labels = first_col.astype(str).tolist()
            feats = df.iloc[:, 1:].values.astype(np.float64)
        else:
            # All numeric — use row indices as labels
            labels = [str(i) for i in range(len(df))]
            feats = df.values.astype(np.float64)
    return feats, labels


def load_features(path: str) -> tuple[np.ndarray, list[str]]:
    """Load protein features from a PLM folder or CSV file.

    Parameters
    ----------
    path : str
        Path to a folder containing ``*.pt`` PLM embedding files,
        or a ``.csv`` feature matrix.

    Returns
    -------
    feats : (N, D) float64 array
    labels : list of N protein name strings
    """
    p = Path(path)
    if p.is_dir():
        from data import prepare_embedding_data
        feats_tensor, labels_arr = prepare_embedding_data(str(p), withroot=False, fmt=".pt")
        return feats_tensor.numpy().astype(np.float64), [str(l) for l in labels_arr]
    elif p.suffix.lower() == ".csv":
        return _load_csv_features(p)
    else:
        raise ValueError(
            f"Cannot load features from {path!r}: "
            "expected a directory (PLM *.pt files) or a .csv file"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Insert new proteins into an existing Poincaré map",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Feature inputs")[0].strip(),
    )

    io = parser.add_argument_group("I/O")
    io.add_argument(
        "--embedding_csv", required=True,
        help="Existing Poincaré map CSV with columns pm1, pm2, <protein_id>",
    )
    io.add_argument(
        "--existing_features", required=True,
        help="Features for every protein already in the map (PLM folder or CSV)",
    )
    io.add_argument(
        "--new_features", required=True,
        help="Features for the proteins to insert (PLM folder or CSV)",
    )
    io.add_argument(
        "--output_csv", default=None,
        help="Output CSV path (default: <embedding_csv stem>_inserted.csv)",
    )

    hp = parser.add_argument_group("Hyperparameters (must match map training values)")
    hp.add_argument("--gamma",    type=float, default=2.0,    help="Kernel bandwidth (default 2.0)")
    hp.add_argument("--knn",      type=int,   default=5,      help="Number of anchor neighbours (default 5)")
    hp.add_argument("--distlocal", type=str,  default="cosine",
                    help="Feature-space distance metric (default cosine)")

    ins = parser.add_argument_group("Insertion options")
    ins.add_argument(
        "--strategy",
        choices=["independent", "seq_center_first", "seq_peri_first", "iterative"],
        default="independent",
        help="Insertion strategy (default: independent — fastest, equivalent quality)",
    )
    ins.add_argument(
        "--max_iters", type=int, default=15,
        help="Max refinement rounds for the iterative strategy (default 15)",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # Load existing embedding
    # ------------------------------------------------------------------
    logger.info("Loading embedding from %s", args.embedding_csv)
    emb_df = pd.read_csv(args.embedding_csv)

    missing_cols = {"pm1", "pm2"} - set(emb_df.columns)
    if missing_cols:
        raise ValueError(f"embedding_csv is missing columns: {missing_cols}")

    label_col = emb_df.columns[-1]   # last column = protein id
    existing_labels: list[str] = emb_df[label_col].astype(str).tolist()
    existing_embs: np.ndarray  = emb_df[["pm1", "pm2"]].values.astype(np.float64)
    logger.info("Loaded %d existing proteins", len(existing_labels))

    # ------------------------------------------------------------------
    # Load existing features and align to embedding order
    # ------------------------------------------------------------------
    logger.info("Loading existing features from %s", args.existing_features)
    all_feats, all_labels = load_features(args.existing_features)
    label_to_feat: dict[str, np.ndarray] = dict(zip(all_labels, all_feats))

    missing_in_feats = [l for l in existing_labels if l not in label_to_feat]
    if missing_in_feats:
        raise ValueError(
            f"Features missing for {len(missing_in_feats)} proteins "
            f"(first 5: {missing_in_feats[:5]}). "
            "Ensure --existing_features covers all proteins in --embedding_csv."
        )
    existing_feats = np.vstack([label_to_feat[l] for l in existing_labels])
    logger.info("Aligned feature matrix: %s", existing_feats.shape)

    # ------------------------------------------------------------------
    # Load new proteins
    # ------------------------------------------------------------------
    logger.info("Loading new protein features from %s", args.new_features)
    new_feats, new_labels = load_features(args.new_features)
    logger.info("New proteins to insert: %d", len(new_labels))

    overlap = set(existing_labels) & set(new_labels)
    if overlap:
        logger.warning(
            "%d protein(s) in --new_features already appear in the map: %s",
            len(overlap), sorted(overlap)[:5],
        )

    # ------------------------------------------------------------------
    # Insert
    # ------------------------------------------------------------------
    from insertion import insert_proteins

    positions, elapsed = insert_proteins(
        new_feats=new_feats,
        existing_feats=existing_feats,
        existing_embs=existing_embs,
        gamma=args.gamma,
        knn=args.knn,
        distlocal=args.distlocal,
        strategy=args.strategy,
        max_iters=args.max_iters,
    )

    # ------------------------------------------------------------------
    # Build and save output CSV
    # ------------------------------------------------------------------
    new_rows = pd.DataFrame(
        {
            "pm1": positions[:, 0],
            "pm2": positions[:, 1],
            label_col: new_labels,
        }
    )
    result_df = pd.concat([emb_df, new_rows], ignore_index=True)

    out_path = args.output_csv
    if out_path is None:
        stem = Path(args.embedding_csv).stem
        out_path = str(Path(args.embedding_csv).parent / f"{stem}_inserted.csv")

    result_df.to_csv(out_path, index=False)
    logger.info(
        "Saved %d proteins (%d original + %d inserted) to %s",
        len(result_df), len(emb_df), len(new_rows), out_path,
    )


if __name__ == "__main__":
    main()
