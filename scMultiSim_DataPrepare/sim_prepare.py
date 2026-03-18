import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def read_matrix(file_path: Path) -> pd.DataFrame:
    return pd.read_csv(file_path, index_col=0)


def cluster_matrices(folder_path: Path, n_clusters: int) -> dict:
    file_list = sorted([p for p in folder_path.iterdir() if p.suffix == ".csv"])
    if not file_list:
        raise FileNotFoundError(f"No CSV files found in {folder_path}")

    filenames = []
    all_grn_array = []
    for file_path in file_list:
        matrix = read_matrix(file_path).values
        all_grn_array.append(matrix.flatten())
        filenames.append(file_path.stem)

    all_grn_array = np.array(all_grn_array)
    print("GRN matrix stack shape:", all_grn_array.shape)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
    cluster_ids = kmeans.fit_predict(all_grn_array)
    return dict(zip(filenames, cluster_ids))


def find_cell_loc(base_dir: Path) -> pd.DataFrame | None:
    cell_loc_path = base_dir / "cell_loc.csv"
    if cell_loc_path.exists():
        return pd.read_csv(cell_loc_path, index_col=0)

    expr_layout_path = base_dir / "expression_loc_cluster_wlayout.csv"
    if expr_layout_path.exists():
        expr_df = pd.read_csv(expr_layout_path, index_col=0)
        if {"x", "y"}.issubset(expr_df.columns):
            # Keep only location columns when using expression file as a fallback.
            return expr_df[["x", "y"]]

    return None


def average_grn(cell_grn_dir: Path) -> pd.DataFrame:
    file_list = sorted([p for p in cell_grn_dir.iterdir() if p.suffix == ".csv"])
    if not file_list:
        raise FileNotFoundError(f"No CSV files found in {cell_grn_dir}")

    sum_matrix = None
    index = None
    columns = None
    for file_path in file_list:
        df = read_matrix(file_path)
        mat = df.values
        if sum_matrix is None:
            sum_matrix = np.zeros_like(mat, dtype=float)
            index = [str(i) for i in df.index]
            columns = [str(c) for c in df.columns]
        sum_matrix += mat

    avg_matrix = sum_matrix / len(file_list)
    return pd.DataFrame(avg_matrix, index=index, columns=columns)


def non_zero_gene_pairs(df: pd.DataFrame) -> pd.DataFrame:
    nz = df.stack().reset_index()
    nz.columns = ["gene1", "gene2", "interaction_strength"]
    nz = nz[nz["interaction_strength"] != 0]
    out = nz[["gene1", "gene2"]].astype(str)
    return out


def normalize_counts(df: pd.DataFrame, scale_factor=10000, pseudo_count=1) -> pd.DataFrame:
    total_counts = df.sum(axis=1)
    normalized_df = df.div(total_counts, axis=0) * scale_factor
    normalized_df = np.log1p(normalized_df + pseudo_count)
    return normalized_df


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare scMultiSim output for SVGRN input.")
    parser.add_argument("--folder-name", default="example_data", help="Folder under in_sim.")
    parser.add_argument("--n-clusters", type=int, default=4, help="Number of GRN clusters.")
    parser.add_argument(
        "--gene-number",
        type=int,
        default=None,
        help="Optional fixed gene number for normalized_count columns (1..N).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    base_dir = Path("in_sim") / args.folder_name
    cell_grn_dir = base_dir / "cell_specific_GRN"
    if not cell_grn_dir.exists():
        raise FileNotFoundError(f"Missing directory: {cell_grn_dir}")

    # 1) GRN clustering
    cluster_dict = cluster_matrices(cell_grn_dir, args.n_clusters)
    grn_cluster_ids_path = base_dir / "GRN_cluster_ids.csv"
    pd.DataFrame.from_dict(cluster_dict, orient="index", columns=["ClusterID"]).to_csv(grn_cluster_ids_path)
    print(f"Saved {grn_cluster_ids_path}")

    # 2) Build cell location + GRN cluster table
    cell_loc_df = find_cell_loc(base_dir)
    cell_loc_cluster_path = base_dir / f"cell_loc_GRNcluster{args.n_clusters}.csv"
    if cell_loc_df is None:
        print("Skip cell_loc_GRNcluster: neither cell_loc.csv nor expression_loc_cluster_wlayout.csv has x/y.")
    else:
        grn_cluster_ids_df = pd.read_csv(grn_cluster_ids_path, index_col=0)
        combined_df = cell_loc_df.join(grn_cluster_ids_df)
        combined_df.to_csv(cell_loc_cluster_path)
        print(f"Saved {cell_loc_cluster_path}")

    # 3) Average GRN over cells
    avg_df = average_grn(cell_grn_dir)
    allcells_grn_path = base_dir / "allcells_GRN.csv"
    avg_df.to_csv(allcells_grn_path)
    print(f"Saved {allcells_grn_path}")

    # 4) Non-zero pairs for tissue-level GRN
    allcells_pairs_path = base_dir / "allcells_gt_gene_pairs.csv"
    non_zero_gene_pairs(avg_df).to_csv(allcells_pairs_path, index=False)
    print(f"Saved {allcells_pairs_path}")

    # 5) Non-zero pairs for each cell-specific GRN
    cell_pairs_dir = base_dir / "cell_specific_gt_gene_pair"
    cell_pairs_dir.mkdir(exist_ok=True)
    for file_path in sorted([p for p in cell_grn_dir.iterdir() if p.suffix == ".csv"]):
        df = read_matrix(file_path)
        out = non_zero_gene_pairs(df)
        out.to_csv(cell_pairs_dir / f"{file_path.stem}_gt_gene_pairs.csv", index=False)
    print(f"Saved per-cell GT pairs to {cell_pairs_dir}")

    # 6) Normalize raw count if raw_count.csv exists
    raw_count_path = base_dir / "raw_count.csv"
    normalized_count_path = base_dir / "normalized_count.csv"
    if raw_count_path.exists():
        raw_df = pd.read_csv(raw_count_path, index_col=0)
        normalized_df = normalize_counts(raw_df)
        if args.gene_number is not None:
            normalized_df.columns = [str(i) for i in range(1, args.gene_number + 1)]
        normalized_df.to_csv(normalized_count_path)
        print(f"Saved {normalized_count_path}")
    else:
        print("Skip normalization: raw_count.csv not found.")

    # 7) Combine normalized expression + location/cluster if both exist
    if normalized_count_path.exists() and cell_loc_cluster_path.exists():
        gene_exp_df = pd.read_csv(normalized_count_path, index_col=0)
        loc_cluster_df = pd.read_csv(cell_loc_cluster_path, index_col=0)
        combined_df = gene_exp_df.join(loc_cluster_df)
        out_expr = base_dir / "expression_loc_cluster_wlayout.csv"
        combined_df.to_csv(out_expr)
        print(f"Saved {out_expr}")
    else:
        print("Skip expression_loc_cluster_wlayout combine step (missing normalized_count or cell_loc_GRNcluster).")


if __name__ == "__main__":
    main()
