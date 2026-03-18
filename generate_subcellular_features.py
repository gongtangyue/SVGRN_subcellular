import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.subcellular_colocalization import compute_gaussian_colocalization
from src.subcellular_feature_store import (
    min_max_scale,
    make_subcellular_feature_config,
    save_subcellular_feature_artifacts,
)
from src.subcellular_grid import build_gene_grid_from_subcellular_csv, normalize_gene_grid


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Precompute shared subcellular features for a dataset so both allcell "
            "and singlecell training can load them directly."
        )
    )
    parser.add_argument(
        "dataset_dir",
        type=Path,
        help="Dataset directory under in_sim, e.g. in_sim/g110_c2k_n01",
    )
    parser.add_argument(
        "--expr-file",
        type=Path,
        default=None,
        help="Expression CSV path. Defaults to <dataset_dir>/expression_loc_cluster_wlayout.csv",
    )
    parser.add_argument(
        "--subcellular-data-dir",
        type=Path,
        default=None,
        help="Directory with per-cell *_subcellular.csv files. Defaults to <dataset_dir>/subcellular_data",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for generated feature files. Defaults to <dataset_dir>/subcellular_features",
    )
    parser.add_argument("--subcell-sigma", type=float, default=0.03)
    parser.add_argument("--subcell-grid-size", type=int, default=16)
    parser.add_argument("--subcell-r-cell", type=float, default=0.05)
    parser.add_argument("--subcell-splat-sigma", type=float, default=1.0)
    parser.add_argument("--subcell-splat-radius", type=int, default=1)
    parser.add_argument("--subcell-grid-norm", type=str, default="log1p")
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of worker processes used to parallelize cells. 1 means serial.",
    )
    return parser.parse_args()


def _load_subcellular_csv(csv_path):
    if not csv_path.exists():
        return None
    sub_df = pd.read_csv(csv_path)
    if sub_df.empty:
        return None
    return sub_df


def _process_single_cell(task):
    (
        cell_id,
        cell_x,
        cell_y,
        csv_path,
        gene_names,
        subcell_sigma,
        subcell_grid_size,
        subcell_r_cell,
        subcell_splat_sigma,
        subcell_splat_radius,
        subcell_grid_norm,
    ) = task

    sub_df = _load_subcellular_csv(csv_path)
    if sub_df is None:
        coloc = np.zeros((len(gene_names), len(gene_names)), dtype=np.float32)
        grid = np.zeros(
            (len(gene_names), subcell_grid_size, subcell_grid_size),
            dtype=np.float32,
        )
        return cell_id, coloc, grid, 1

    coloc = compute_gaussian_colocalization(sub_df, gene_names, subcell_sigma).astype(
        np.float32,
        copy=False,
    )
    grid = build_gene_grid_from_subcellular_csv(
        sub_df=sub_df,
        gene_list=gene_names,
        cell_x=cell_x,
        cell_y=cell_y,
        grid_size=subcell_grid_size,
        r_cell=subcell_r_cell,
        splat_sigma=subcell_splat_sigma,
        radius=subcell_splat_radius,
    )
    grid = normalize_gene_grid(grid, mode=subcell_grid_norm).astype(
        np.float32,
        copy=False,
    )
    return cell_id, coloc, grid, 0


def generate_dataset_features(
    expr_df,
    subcellular_data_dir,
    subcell_sigma,
    subcell_grid_size,
    subcell_r_cell,
    subcell_splat_sigma,
    subcell_splat_radius,
    subcell_grid_norm,
    progress_every,
    num_workers,
):
    cell_ids = expr_df.index.astype(str)
    gene_names = [str(col) for col in expr_df.columns if col not in ("x", "y", "ClusterID")]
    n_cells = len(cell_ids)
    coloc_features = [None] * n_cells
    grid_features = [None] * n_cells
    missing_or_empty = 0
    tasks = [
        (
            str(cell_id),
            float(row["x"]),
            float(row["y"]),
            subcellular_data_dir / f"{cell_id}_subcellular.csv",
            gene_names,
            subcell_sigma,
            subcell_grid_size,
            subcell_r_cell,
            subcell_splat_sigma,
            subcell_splat_radius,
            subcell_grid_norm,
        )
        for cell_id, row in expr_df.iterrows()
    ]

    worker_count = max(1, int(num_workers))
    executor = None
    executor_kind = "serial"
    if worker_count == 1:
        iterator = map(_process_single_cell, tasks)
    else:
        max_workers = min(worker_count, os.cpu_count() or worker_count)
        try:
            executor = ProcessPoolExecutor(max_workers=max_workers)
            iterator = executor.map(_process_single_cell, tasks, chunksize=1)
            executor_kind = f"processes={max_workers}"
        except (OSError, PermissionError) as exc:
            print(
                f"ProcessPool unavailable ({exc}). Falling back to ThreadPoolExecutor "
                f"with {max_workers} workers."
            )
            executor = ThreadPoolExecutor(max_workers=max_workers)
            iterator = executor.map(_process_single_cell, tasks)
            executor_kind = f"threads={max_workers}"

    print(f"Feature generation parallelism: {executor_kind}")

    try:
        for idx, (cell_id, coloc, grid, missing_flag) in enumerate(iterator, start=1):
            array_idx = idx - 1
            coloc_features[array_idx] = coloc
            grid_features[array_idx] = grid
            missing_or_empty += missing_flag

            if progress_every > 0 and (idx % progress_every == 0 or idx == n_cells):
                print(f"[{idx}/{n_cells}] processed {cell_id}")
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    subcell_coloc = np.stack(coloc_features, axis=0).astype(np.float32, copy=False)
    subcell_grid = np.stack(grid_features, axis=0).astype(np.float32, copy=False)
    subcell_coloc = min_max_scale(subcell_coloc.reshape(len(cell_ids), -1))
    subcell_coloc = subcell_coloc.reshape(len(cell_ids), len(gene_names), len(gene_names))
    return cell_ids, gene_names, subcell_coloc, subcell_grid, missing_or_empty


def main():
    args = parse_args()

    dataset_dir = args.dataset_dir.expanduser().resolve()
    expr_file = (
        args.expr_file.expanduser().resolve()
        if args.expr_file is not None
        else dataset_dir / "expression_loc_cluster_wlayout.csv"
    )
    subcellular_data_dir = (
        args.subcellular_data_dir.expanduser().resolve()
        if args.subcellular_data_dir is not None
        else dataset_dir / "subcellular_data"
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else dataset_dir / "subcellular_features"
    )

    if not expr_file.exists():
        raise FileNotFoundError(f"Expression file not found: {expr_file}")
    if not subcellular_data_dir.exists():
        raise FileNotFoundError(f"Subcellular data directory not found: {subcellular_data_dir}")

    expr_df = pd.read_csv(expr_file, index_col=0)
    expr_df.index = expr_df.index.astype(str)
    expr_df.columns = expr_df.columns.astype(str)

    missing_cols = [col for col in ("x", "y") if col not in expr_df.columns]
    if missing_cols:
        raise ValueError(f"Expression file missing required columns: {missing_cols}")

    config = make_subcellular_feature_config(
        subcell_sigma=args.subcell_sigma,
        subcell_grid_size=args.subcell_grid_size,
        subcell_r_cell=args.subcell_r_cell,
        subcell_splat_sigma=args.subcell_splat_sigma,
        subcell_splat_radius=args.subcell_splat_radius,
        subcell_grid_norm=args.subcell_grid_norm,
    )

    print(f"Dataset dir: {dataset_dir}")
    print(f"Expression file: {expr_file}")
    print(f"Subcellular CSV dir: {subcellular_data_dir}")
    print(f"Output dir: {output_dir}")

    cell_ids, gene_names, subcell_coloc, subcell_grid, missing_or_empty = generate_dataset_features(
        expr_df=expr_df,
        subcellular_data_dir=subcellular_data_dir,
        subcell_sigma=args.subcell_sigma,
        subcell_grid_size=args.subcell_grid_size,
        subcell_r_cell=args.subcell_r_cell,
        subcell_splat_sigma=args.subcell_splat_sigma,
        subcell_splat_radius=args.subcell_splat_radius,
        subcell_grid_norm=args.subcell_grid_norm,
        progress_every=args.progress_every,
        num_workers=args.num_workers,
    )

    save_subcellular_feature_artifacts(
        output_dir=output_dir,
        cell_ids=cell_ids,
        gene_names=gene_names,
        subcell_coloc_array=subcell_coloc,
        subcell_grid_array=subcell_grid,
        config=config,
    )

    print(
        "Saved shared subcellular features. "
        f"cells={len(cell_ids)}, genes={len(gene_names)}, "
        f"coloc_shape={subcell_coloc.shape}, grid_shape={subcell_grid.shape}, "
        f"missing_or_empty={missing_or_empty}/{len(cell_ids)}"
    )


if __name__ == "__main__":
    main()
