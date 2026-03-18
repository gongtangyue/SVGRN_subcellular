import argparse
from pathlib import Path

import numpy as np
import pandas as pd


rng = np.random.default_rng(0)


def softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


def sample_counts(expr_values, mode="poisson", count_scale=20.0):
    if mode == "poisson":
        lam = np.clip(expr_values * count_scale, 0, None)
        return rng.poisson(lam)
    if mode == "round":
        return np.rint(np.clip(expr_values * count_scale, 0, None)).astype(int)
    raise ValueError(f"Unknown mode: {mode}")


def sample_uniform_in_disk(center, radius, n):
    if n <= 0:
        return np.empty((0, 2))
    u = rng.random(n)
    v = rng.random(n)
    r = np.sqrt(u) * radius
    theta = 2 * np.pi * v
    x = center[0] + r * np.cos(theta)
    y = center[1] + r * np.sin(theta)
    return np.stack([x, y], axis=1)


def u_gene_bias(points, center, r_nuc, u_nuc=0.5, u_cyt=0.0):
    d = points - center
    in_nuc = (d[:, 0] ** 2 + d[:, 1] ** 2) <= r_nuc ** 2
    return np.where(in_nuc, u_nuc, u_cyt)


def build_hotspots(center, r_nuc, K=2):
    return sample_uniform_in_disk(center, r_nuc, K)


def eval_field(points, mu, sigma, amp, bias):
    m = points.shape[0]
    out = np.full(m, bias, dtype=float)
    for k in range(mu.shape[0]):
        diff = points - mu[k]
        d2 = diff[:, 0] ** 2 + diff[:, 1] ** 2
        out += amp * np.exp(-d2 / (2 * sigma ** 2))
    return out


def sample_points_by_lambda(cand, lam, n):
    if n <= 0:
        return np.empty((0, 2))
    w = np.clip(lam, 0, None)
    s = w.sum()
    if s <= 0:
        idx = rng.integers(0, cand.shape[0], size=n)
        return cand[idx]
    p = w / s
    idx = rng.choice(cand.shape[0], size=n, replace=True, p=p)
    return cand[idx]


def matrix_to_target_regulators(matrix_df, gene_cols):
    genes = set(gene_cols)
    matrix_df.index = matrix_df.index.map(str)
    matrix_df.columns = matrix_df.columns.map(str)

    target_to_regs = {}
    row_genes = [g for g in matrix_df.index if g in genes]
    col_genes = [g for g in matrix_df.columns if g in genes]
    if not row_genes or not col_genes:
        return target_to_regs

    sub = matrix_df.loc[row_genes, col_genes]
    for target in sub.columns:
        vals = sub[target]
        regs = vals.index[vals.to_numpy(dtype=float) != 0].tolist()
        if regs:
            target_to_regs[str(target)] = [str(r) for r in regs]
    return target_to_regs


def edge_df_to_target_regulators(edges_df, gene_cols):
    genes = set(gene_cols)
    required = {"gene1", "gene2"}
    if not required.issubset(edges_df.columns):
        raise ValueError("Edge table must contain columns: gene1, gene2")

    edges_df = edges_df[
        edges_df["gene1"].astype(str).isin(genes)
        & edges_df["gene2"].astype(str).isin(genes)
    ]
    return edges_df.groupby("gene2")["gene1"].apply(lambda x: [str(v) for v in x]).to_dict()


def load_target_regulators(cell_id, grn_path, gene_cols):
    grn_path = Path(grn_path)

    if grn_path.is_dir():
        candidates = [
            grn_path / f"{cell_id}.csv",
            grn_path / f"{cell_id}_gt_gene_pairs.csv",
        ]
        file_path = next((p for p in candidates if p.exists()), None)
        if file_path is None:
            return {}

        df = pd.read_csv(file_path, index_col=0)
        if {"gene1", "gene2"}.issubset(df.columns):
            return edge_df_to_target_regulators(df, gene_cols)
        return matrix_to_target_regulators(df, gene_cols)

    raise FileNotFoundError(f"GRN directory not found: {grn_path}")


def generate_one_cell_transcripts(
    cell_id,
    row,
    gene_cols,
    grn_path,
    mode="poisson",
    count_scale=20.0,
    r_cell=0.05,
    r_nuc=0.025,
    m_candidates=40000,
    k_hot=2,
    sigma_reg=0.02,
    b_reg=0.01,
    beta0=-1.0,
):
    expr_values = row[gene_cols].to_numpy(dtype=float)
    counts = sample_counts(expr_values, mode=mode, count_scale=count_scale)

    center = row[["x", "y"]].to_numpy(dtype=float)
    target_to_regs = load_target_regulators(str(cell_id), grn_path, gene_cols)

    cand = sample_uniform_in_disk(center, r_cell, m_candidates)
    u = u_gene_bias(cand, center, r_nuc, u_nuc=0.5, u_cyt=0.0)

    regs_in_cell = sorted({r for regs in target_to_regs.values() for r in regs})
    f_reg = {}
    for reg in regs_in_cell:
        amp = float(max(row.get(reg, 0.0), 0.0)) + 1e-6
        mu = build_hotspots(center, r_nuc, K=k_hot)
        f_reg[reg] = eval_field(cand, mu, sigma=sigma_reg, amp=amp, bias=b_reg)

    records = []
    for gene, n in zip(gene_cols, counts):
        n = int(n)
        if n <= 0:
            continue

        regs = target_to_regs.get(str(gene), [])

        grn_term = np.zeros(cand.shape[0], dtype=float)
        for reg in regs:
            if reg in f_reg:
                grn_term += f_reg[reg]

        lam = softplus(beta0 + grn_term + u)
        pts = sample_points_by_lambda(cand, lam, n)
        records.append(pd.DataFrame({"gene": str(gene), "x": pts[:, 0], "y": pts[:, 1]}))

    if records:
        return pd.concat(records, ignore_index=True)
    return pd.DataFrame(columns=["gene", "x", "y"])


def generate_subcellular_transcripts(
    expr_path,
    grn_path,
    mode="poisson",
    count_scale=20.0,
    r_cell=0.05,
    r_nuc=0.025,
    m_candidates=40000,
    k_hot=2,
    sigma_reg=0.02,
    b_reg=0.01,
    beta0=-1.0,
):
    expr_df = pd.read_csv(expr_path, index_col=0)
    expr_df.columns = expr_df.columns.map(str)
    gene_cols = [c for c in expr_df.columns if c not in ("x", "y", "ClusterID")]

    missing_cols = [c for c in ("x", "y") if c not in expr_df.columns]
    if missing_cols:
        raise ValueError(f"Expression file missing required columns: {missing_cols}")

    cell_transcripts = {}

    for cell_id, row in expr_df.iterrows():
        cell_transcripts[str(cell_id)] = generate_one_cell_transcripts(
            cell_id=cell_id,
            row=row,
            gene_cols=gene_cols,
            grn_path=grn_path,
            mode=mode,
            count_scale=count_scale,
            r_cell=r_cell,
            r_nuc=r_nuc,
            m_candidates=m_candidates,
            k_hot=k_hot,
            sigma_reg=sigma_reg,
            b_reg=b_reg,
            beta0=beta0,
        )

    return cell_transcripts


def generate_and_save_subcellular_transcripts(
    expr_path,
    grn_path,
    output_dir,
    mode="poisson",
    count_scale=20.0,
    r_cell=0.05,
    r_nuc=0.025,
    m_candidates=40000,
    k_hot=2,
    sigma_reg=0.02,
    b_reg=0.01,
    beta0=-1.0,
    progress_every=1,
    limit_cells=None,
):
    expr_df = pd.read_csv(expr_path, index_col=0)
    expr_df.columns = expr_df.columns.map(str)
    gene_cols = [c for c in expr_df.columns if c not in ("x", "y", "ClusterID")]

    missing_cols = [c for c in ("x", "y") if c not in expr_df.columns]
    if missing_cols:
        raise ValueError(f"Expression file missing required columns: {missing_cols}")

    total_cells = len(expr_df)
    max_cells = total_cells if limit_cells is None else min(total_cells, int(limit_cells))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Start generating {max_cells} cells -> {output_dir}")

    first_cell_id = None
    for idx, (cell_id, row) in enumerate(expr_df.iterrows(), start=1):
        if idx > max_cells:
            break

        df = generate_one_cell_transcripts(
            cell_id=cell_id,
            row=row,
            gene_cols=gene_cols,
            grn_path=grn_path,
            mode=mode,
            count_scale=count_scale,
            r_cell=r_cell,
            r_nuc=r_nuc,
            m_candidates=m_candidates,
            k_hot=k_hot,
            sigma_reg=sigma_reg,
            b_reg=b_reg,
            beta0=beta0,
        )
        df.to_csv(output_dir / f"{cell_id}_subcellular.csv", index=False)

        if first_cell_id is None:
            first_cell_id = str(cell_id)
        if progress_every > 0 and (idx % progress_every == 0 or idx == max_cells):
            print(f"[{idx}/{max_cells}] saved {cell_id}_subcellular.csv")

    return max_cells, first_cell_id


def parse_args():
    default_expr = "E:/st/SVGRN_subcellular/in_sim/g110_c2k_n01/expression_loc_cluster_wlayout.csv"
    default_grn = "E:/st/SVGRN_subcellular/in_sim/g110_c2k_n01/cell_specific_GRN"

    parser = argparse.ArgumentParser(description="Generate subcellular transcript coordinates.")
    parser.add_argument("--expr-path", type=Path, default=default_expr, help="Expression CSV path.")
    parser.add_argument(
        "--grn-path",
        type=Path,
        default=default_grn,
        help="GRN directory containing per-cell files (e.g., cell1.csv).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(r"E:\\st\\SVGRN_subcellular\\in_sim\\g110_c2k_n01\\subcellular_data"),
        help="Optional output directory for per-cell transcript CSV files.",
    )
    parser.add_argument("--mode", type=str, default="poisson", choices=["poisson", "round"])
    parser.add_argument("--count-scale", type=float, default=20.0)
    parser.add_argument("--r-cell", type=float, default=0.05)
    parser.add_argument("--r-nuc", type=float, default=0.025)
    parser.add_argument("--m-candidates", type=int, default=40000)
    parser.add_argument("--k-hot", type=int, default=2)
    parser.add_argument("--sigma-reg", type=float, default=0.02)
    parser.add_argument("--b-reg", type=float, default=0.01)
    parser.add_argument("--beta0", type=float, default=-1.0)
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument("--limit-cells", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.output_dir is not None:
        saved_cells, first_cell = generate_and_save_subcellular_transcripts(
            expr_path=args.expr_path,
            grn_path=args.grn_path,
            output_dir=args.output_dir,
            mode=args.mode,
            count_scale=args.count_scale,
            r_cell=args.r_cell,
            r_nuc=args.r_nuc,
            m_candidates=args.m_candidates,
            k_hot=args.k_hot,
            sigma_reg=args.sigma_reg,
            b_reg=args.b_reg,
            beta0=args.beta0,
            progress_every=args.progress_every,
            limit_cells=args.limit_cells,
        )
        if saved_cells <= 0:
            print("No cells found.")
            return
        print(f"Saved {saved_cells} cells to {args.output_dir}")
        if first_cell is not None:
            example_path = args.output_dir / f"{first_cell}_subcellular.csv"
            print(f"example cell: {first_cell}")
            print(pd.read_csv(example_path).head())
        return

    cell_transcripts = generate_subcellular_transcripts(
        expr_path=args.expr_path,
        grn_path=args.grn_path,
        mode=args.mode,
        count_scale=args.count_scale,
        r_cell=args.r_cell,
        r_nuc=args.r_nuc,
        m_candidates=args.m_candidates,
        k_hot=args.k_hot,
        sigma_reg=args.sigma_reg,
        b_reg=args.b_reg,
        beta0=args.beta0,
    )
    if not cell_transcripts:
        print("No cells found.")
        return
    example_cell = next(iter(cell_transcripts))
    print(f"example cell: {example_cell}")
    print(cell_transcripts[example_cell].head())


if __name__ == "__main__":
    main()
