import numpy as np
import pandas as pd


def _normalize_local_coords(sub_df, cell_x, cell_y, r_cell):
    dx = (sub_df["x"].to_numpy(dtype=np.float32) - float(cell_x)) / float(r_cell)
    dy = (sub_df["y"].to_numpy(dtype=np.float32) - float(cell_y)) / float(r_cell)
    return dx, dy


def _gaussian_patch_weights(fx, fy, radius=1, sigma=1.0):
    cx = int(np.floor(fx))
    cy = int(np.floor(fy))

    coords = []
    weights = []
    for yy in range(cy - radius, cy + radius + 1):
        for xx in range(cx - radius, cx + radius + 1):
            d2 = (fx - xx) ** 2 + (fy - yy) ** 2
            weight = np.exp(-d2 / (2.0 * sigma * sigma))
            coords.append((yy, xx))
            weights.append(weight)

    weights = np.asarray(weights, dtype=np.float32)
    total = weights.sum()
    if total > 0:
        weights /= total
    return coords, weights


def build_gene_grid_from_subcellular_csv(
    sub_df: pd.DataFrame,
    gene_list,
    cell_x,
    cell_y,
    grid_size=16,
    r_cell=0.05,
    splat_sigma=1.0,
    radius=1,
):
    n_gene = len(gene_list)
    gene_to_idx = {str(gene): idx for idx, gene in enumerate(gene_list)}
    grid = np.zeros((n_gene, grid_size, grid_size), dtype=np.float32)
    required_cols = {"gene", "x", "y"}

    if sub_df is None or sub_df.empty or not required_cols.issubset(sub_df.columns):
        return grid

    filtered = sub_df.copy()
    filtered["gene"] = filtered["gene"].astype(str)
    filtered = filtered[filtered["gene"].isin(gene_to_idx)]
    filtered = filtered.dropna(subset=["x", "y"])
    if filtered.empty:
        return grid

    dx, dy = _normalize_local_coords(filtered, cell_x, cell_y, r_cell)
    gx = (dx + 1.0) * 0.5 * (grid_size - 1)
    gy = (dy + 1.0) * 0.5 * (grid_size - 1)
    genes = filtered["gene"].tolist()

    for gene, fx, fy in zip(genes, gx, gy):
        if fx < 0 or fx > (grid_size - 1) or fy < 0 or fy > (grid_size - 1):
            continue

        gene_idx = gene_to_idx[gene]
        coords, weights = _gaussian_patch_weights(
            fx,
            fy,
            radius=radius,
            sigma=splat_sigma,
        )

        for (yy, xx), weight in zip(coords, weights):
            if 0 <= yy < grid_size and 0 <= xx < grid_size:
                grid[gene_idx, yy, xx] += weight

    return grid


def normalize_gene_grid(grid, mode="log1p"):
    if mode == "log1p":
        return np.log1p(grid).astype(np.float32)

    if mode == "per_gene_sum":
        out = grid.copy()
        denom = out.reshape(out.shape[0], -1).sum(axis=1, keepdims=True)
        denom[denom == 0] = 1.0
        out = out.reshape(out.shape[0], -1) / denom
        return out.reshape(grid.shape).astype(np.float32)

    if mode == "none":
        return grid.astype(np.float32)

    raise ValueError(f"Unknown normalization mode: {mode}")
