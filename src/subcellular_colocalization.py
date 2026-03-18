import numpy as np


def compute_gaussian_colocalization(sub_df, gene_list, sigma):
    gene_to_idx = {str(gene): idx for idx, gene in enumerate(gene_list)}
    n_gene = len(gene_list)
    colocalization = np.zeros((n_gene, n_gene), dtype=np.float32)
    required_cols = {"gene", "x", "y"}

    if sigma <= 0 or sub_df is None or sub_df.empty or not required_cols.issubset(sub_df.columns):
        return colocalization

    filtered = sub_df.loc[:, ["gene", "x", "y"]].copy()
    filtered["gene"] = filtered["gene"].astype(str)
    filtered = filtered[filtered["gene"].isin(gene_to_idx)]
    filtered = filtered.dropna(subset=["x", "y"])
    if filtered.shape[0] < 2:
        return colocalization

    coords = filtered[["x", "y"]].to_numpy(dtype=np.float32)
    gene_idx = filtered["gene"].map(gene_to_idx).to_numpy(dtype=np.int64)
    denom = 2.0 * float(sigma) ** 2

    for i in range(coords.shape[0] - 1):
        deltas = coords[i + 1:] - coords[i]
        dist_sq = np.einsum("ij,ij->i", deltas, deltas)
        weights = np.exp(-dist_sq / denom).astype(np.float32, copy=False)
        if weights.size == 0:
            continue

        gi = gene_idx[i]
        gj = gene_idx[i + 1:]
        np.add.at(colocalization, (np.full(gj.shape, gi, dtype=np.int64), gj), weights)
        np.add.at(colocalization, (gj, np.full(gj.shape, gi, dtype=np.int64)), weights)

    total_weight = colocalization.sum()
    if total_weight > 0:
        colocalization /= total_weight

    return colocalization
