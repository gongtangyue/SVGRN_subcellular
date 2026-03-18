import json
from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_DIR_NAME = "subcellular_features"
CELL_IDS_FILE = "cell_ids.npy"
GENE_NAMES_FILE = "gene_names.npy"
COLOCALIZATION_FILE = "subcell_coloc.npy"
GRID_FILE = "subcell_grid.npy"
CONFIG_FILE = "subcell_feature_config.json"


def min_max_scale(df_or_array):
    if isinstance(df_or_array, pd.DataFrame):
        denom = df_or_array.max(0) - df_or_array.min(0)
        denom = denom.replace(0, 1)
        return (df_or_array - df_or_array.min(0)) / denom

    arr = np.asarray(df_or_array, dtype=np.float32)
    if arr.size == 0:
        return arr
    min_vals = arr.min(axis=0, keepdims=True)
    denom = arr.max(axis=0, keepdims=True) - min_vals
    denom[denom == 0] = 1
    return (arr - min_vals) / denom


def make_subcellular_feature_config(
    subcell_sigma=0.03,
    subcell_grid_size=16,
    subcell_r_cell=0.05,
    subcell_splat_sigma=1.0,
    subcell_splat_radius=1,
    subcell_grid_norm="log1p",
):
    return {
        "subcell_sigma": float(subcell_sigma),
        "subcell_grid_size": int(subcell_grid_size),
        "subcell_r_cell": float(subcell_r_cell),
        "subcell_splat_sigma": float(subcell_splat_sigma),
        "subcell_splat_radius": int(subcell_splat_radius),
        "subcell_grid_norm": str(subcell_grid_norm),
    }
def save_subcellular_feature_artifacts(
    output_dir,
    cell_ids,
    gene_names,
    subcell_coloc_array,
    subcell_grid_array,
    config,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / CELL_IDS_FILE, np.asarray(cell_ids, dtype=str))
    np.save(output_dir / GENE_NAMES_FILE, np.asarray(gene_names, dtype=str))
    np.save(output_dir / COLOCALIZATION_FILE, np.asarray(subcell_coloc_array, dtype=np.float32))
    np.save(output_dir / GRID_FILE, np.asarray(subcell_grid_array, dtype=np.float32))
    (output_dir / CONFIG_FILE).write_text(
        json.dumps(config, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _load_required_file(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing precomputed subcellular feature file: {path}")
    return path


def _build_indexer(stored_values, requested_values, value_name):
    index_map = {str(value): idx for idx, value in enumerate(stored_values)}
    indexer = []
    missing = []
    for value in requested_values:
        key = str(value)
        idx = index_map.get(key)
        if idx is None:
            missing.append(key)
        else:
            indexer.append(idx)
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(
            f"Precomputed subcellular features are missing {value_name}(s): {preview}"
        )
    return np.asarray(indexer, dtype=np.int64)
def load_precomputed_subcellular_features(feature_dir, cell_ids, gene_names):
    feature_dir = Path(feature_dir).expanduser().resolve()
    _load_required_file(feature_dir / CONFIG_FILE)

    stored_cell_ids = np.load(_load_required_file(feature_dir / CELL_IDS_FILE), allow_pickle=False)
    stored_gene_names = np.load(_load_required_file(feature_dir / GENE_NAMES_FILE), allow_pickle=False)
    stored_cell_ids = stored_cell_ids.astype(str)
    stored_gene_names = stored_gene_names.astype(str)

    requested_cell_ids = np.asarray(cell_ids, dtype=str)
    requested_gene_names = np.asarray(gene_names, dtype=str)

    cell_indexer = _build_indexer(stored_cell_ids, requested_cell_ids, "cell")
    gene_indexer = _build_indexer(stored_gene_names, requested_gene_names, "gene")

    subcell_coloc = np.load(_load_required_file(feature_dir / COLOCALIZATION_FILE), allow_pickle=False)
    subcell_grid = np.load(_load_required_file(feature_dir / GRID_FILE), allow_pickle=False)

    if subcell_coloc.ndim != 3:
        raise ValueError(
            f"Expected {COLOCALIZATION_FILE} to be 3D [cells, genes, genes], got {subcell_coloc.shape}"
        )
    if subcell_grid.ndim != 4:
        raise ValueError(
            f"Expected {GRID_FILE} to be 4D [cells, genes, grid, grid], got {subcell_grid.shape}"
        )

    subcell_coloc = subcell_coloc[cell_indexer][:, gene_indexer][:, :, gene_indexer]
    subcell_grid = subcell_grid[cell_indexer][:, gene_indexer, :, :]
    subcell_coloc = subcell_coloc.reshape(len(requested_cell_ids), -1).astype(np.float32, copy=False)
    subcell_grid = subcell_grid.astype(np.float32, copy=False)
    return subcell_coloc, subcell_grid, feature_dir
