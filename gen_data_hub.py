import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from gen_data import (
    sample_counts,
    sample_points_by_lambda,
    sample_uniform_in_disk,
)


rng = np.random.default_rng(0)


def matrix_to_target_regulator_weights(matrix_df, gene_cols):
    matrix_df = matrix_df.copy()
    matrix_df.index = matrix_df.index.map(str)
    matrix_df.columns = matrix_df.columns.map(str)

    gene_set = {str(g) for g in gene_cols}
    valid_targets = [g for g in matrix_df.index if g in gene_set]
    valid_regs = [g for g in matrix_df.columns if g in gene_set]
    sub = matrix_df.loc[valid_targets, valid_regs]

    target_to_regs = {}
    for target, vals in sub.iterrows():
        reg_weights = []
        for reg, weight in vals.items():
            weight = float(weight)
            if weight != 0:
                reg_weights.append((str(reg), abs(weight)))
        if reg_weights:
            target_to_regs[str(target)] = reg_weights

    return target_to_regs


def load_target_regulator_weights(cell_id, grn_path, gene_cols):
    grn_path = Path(grn_path)
    if not grn_path.is_dir():
        raise FileNotFoundError(f"GRN directory not found: {grn_path}")

    file_path = grn_path / f"{cell_id}.csv"
    if not file_path.exists():
        return {}

    df = pd.read_csv(file_path, index_col=0)
    return matrix_to_target_regulator_weights(df, gene_cols)


def build_hubs(center, r_nuc, K=3, sigma_reg=0.02, strength_scale=1.0):
    centers = sample_uniform_in_disk(center, r_nuc, K)
    sigma = rng.uniform(0.6 * sigma_reg, 1.4 * sigma_reg, size=K)
    strength = rng.lognormal(mean=0.0, sigma=0.25, size=K) * strength_scale
    return pd.DataFrame(
        {
            "hub": [f"hub{i + 1}" for i in range(K)],
            "x": centers[:, 0],
            "y": centers[:, 1],
            "sigma": sigma,
            "strength": strength,
        }
    )


def assign_targets_to_hubs(target_to_regs, hubs):
    targets = sorted(target_to_regs)
    if not targets:
        return {}

    hub_strength = hubs["strength"].to_numpy(dtype=float)
    if hub_strength.sum() <= 0:
        probs = np.full(len(hubs), 1.0 / len(hubs))
    else:
        probs = hub_strength / hub_strength.sum()

    return {
        target: int(rng.choice(len(hubs), p=probs))
        for target in targets
    }


def tf_hub_weights_from_targets(target_to_regs, target_to_hub, hubs):
    tf_to_weights = {}
    for target, regs in target_to_regs.items():
        hub_idx = target_to_hub.get(str(target))
        if hub_idx is None:
            continue
        for reg, edge_weight in regs:
            weights = tf_to_weights.setdefault(str(reg), np.zeros(len(hubs), dtype=float))
            weights[hub_idx] += float(edge_weight)
    return tf_to_weights


def hub_field(cand, hubs, weights=None, bias=0.0):
    if weights is None:
        weights = np.ones(len(hubs), dtype=float)
    weights = np.asarray(weights, dtype=float)

    out = np.full(cand.shape[0], bias, dtype=float)
    for i, hub in hubs.iterrows():
        weight = weights[i]
        if weight <= 0:
            continue
        dx = cand[:, 0] - float(hub["x"])
        dy = cand[:, 1] - float(hub["y"])
        sigma = float(hub["sigma"])
        amp = float(hub["strength"]) * weight
        out += amp * np.exp(-(dx * dx + dy * dy) / (2 * sigma * sigma))
    return out


def sample_points_from_hubs(cand, hubs, weights, n, top_r=None):
    lam = hub_field(cand, hubs, weights=weights, bias=0.0)
    return sample_points_by_lambda(cand, lam, n, top_r=top_r)


def sample_background_points(cand, n):
    if n <= 0:
        return np.empty((0, 2))
    idx = rng.integers(0, cand.shape[0], size=n)
    return cand[idx]


def target_hub_weights(gene, target_to_hub, hubs):
    weights = np.zeros(len(hubs), dtype=float)
    hub_idx = target_to_hub.get(str(gene))
    if hub_idx is not None:
        weights[hub_idx] = 1.0
    return weights


def generate_one_cell_transcripts_hub(
    cell_id,
    row,
    gene_cols,
    grn_path,
    top_r=None,
    mode="poisson",
    count_scale=20.0,
    r_cell=0.05,
    r_nuc=0.025,
    m_candidates=40000,
    k_hub=3,
    sigma_reg=0.02,
):
    expr_values = row[gene_cols].to_numpy(dtype=float)
    counts = sample_counts(expr_values, mode=mode, count_scale=count_scale)

    center = row[["x", "y"]].to_numpy(dtype=float)
    target_to_regs = load_target_regulator_weights(str(cell_id), grn_path, gene_cols)

    hubs = build_hubs(center, r_nuc, K=k_hub, sigma_reg=sigma_reg)
    target_to_hub = assign_targets_to_hubs(target_to_regs=target_to_regs, hubs=hubs)
    tf_to_hub_weights = tf_hub_weights_from_targets(target_to_regs, target_to_hub, hubs)

    cand = sample_uniform_in_disk(center, r_cell, m_candidates)

    records = []
    target_genes = set(target_to_hub)
    tf_genes = set(tf_to_hub_weights)
    counts_items = [(str(gene), int(n)) for gene, n in zip(gene_cols, counts)]

    # 1) Regulated targets are generated directly from their assigned hub.
    for gene, n in counts_items:
        if n <= 0 or gene not in target_genes or gene in tf_genes:
            continue

        weights = target_hub_weights(gene, target_to_hub, hubs)
        pts = sample_points_from_hubs(cand, hubs, weights, n, top_r=top_r)
        records.append(pd.DataFrame({"gene": gene, "x": pts[:, 0], "y": pts[:, 1]}))

    # 2) TFs are generated after targets, using the hubs of their regulated targets.
    for gene, n in counts_items:
        if n <= 0 or gene not in tf_genes:
            continue

        weights = tf_to_hub_weights[gene]
        pts = sample_points_from_hubs(cand, hubs, weights, n, top_r=top_r)
        records.append(pd.DataFrame({"gene": gene, "x": pts[:, 0], "y": pts[:, 1]}))

    # 3) Genes without a target/TF role are kept as background transcripts.
    for gene, n in counts_items:
        if n <= 0:
            continue
        if gene in target_genes or gene in tf_genes:
            continue

        pts = sample_background_points(cand, n)
        records.append(pd.DataFrame({"gene": gene, "x": pts[:, 0], "y": pts[:, 1]}))

    if records:
        return pd.concat(records, ignore_index=True), hubs, target_to_hub, tf_to_hub_weights
    return pd.DataFrame(columns=["gene", "x", "y"]), hubs, target_to_hub, tf_to_hub_weights


def generate_and_save_subcellular_transcripts_hub(
    expr_path,
    grn_path,
    output_dir,
    top_r=None,
    mode="poisson",
    count_scale=20.0,
    r_cell=0.05,
    r_nuc=0.025,
    m_candidates=40000,
    k_hub=3,
    sigma_reg=0.02,
    save_hub_meta=False,
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
    hub_dir = output_dir / "hub_meta"
    if save_hub_meta:
        hub_dir.mkdir(parents=True, exist_ok=True)

    print(f"Start hub-based generation for {max_cells} cells -> {output_dir}")

    first_cell_id = None
    for idx, (cell_id, row) in enumerate(expr_df.iterrows(), start=1):
        if idx > max_cells:
            break

        df, hubs, target_to_hub, tf_to_hub_weights = generate_one_cell_transcripts_hub(
            cell_id=cell_id,
            row=row,
            gene_cols=gene_cols,
            grn_path=grn_path,
            top_r=top_r,
            mode=mode,
            count_scale=count_scale,
            r_cell=r_cell,
            r_nuc=r_nuc,
            m_candidates=m_candidates,
            k_hub=k_hub,
            sigma_reg=sigma_reg,
        )
        df.to_csv(output_dir / f"{cell_id}_subcellular.csv", index=False)

        if save_hub_meta:
            hubs.to_csv(hub_dir / f"{cell_id}_hubs.csv", index=False)
            pd.DataFrame(
                [{"target": target, "hub": f"hub{hub_idx + 1}"} for target, hub_idx in target_to_hub.items()]
            ).to_csv(hub_dir / f"{cell_id}_target_hubs.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "TF": tf,
                        "hub": f"hub{hub_idx + 1}",
                        "weight": float(weight),
                    }
                    for tf, weights in tf_to_hub_weights.items()
                    for hub_idx, weight in enumerate(weights)
                    if weight > 0
                ]
            ).to_csv(hub_dir / f"{cell_id}_tf_hubs.csv", index=False)

        if first_cell_id is None:
            first_cell_id = str(cell_id)
        if progress_every > 0 and (idx % progress_every == 0 or idx == max_cells):
            print(f"[{idx}/{max_cells}] saved {cell_id}_subcellular.csv")

    return max_cells, first_cell_id


def parse_args():
    default_expr = "E:/st/SVGRN_subcellular/in_sim/g110_c2k_n01/expression_loc_cluster_wlayout.csv"
    default_grn = "E:/st/SVGRN_subcellular/in_sim/g110_c2k_n01/cell_specific_GRN"

    parser = argparse.ArgumentParser(description="Generate hub-based subcellular transcript coordinates.")
    parser.add_argument("--expr-path", type=Path, default=default_expr, help="Expression CSV path.")
    parser.add_argument("--grn-path", type=Path, default=default_grn, help="Per-cell GRN directory.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(r"E:\\st\\SVGRN_subcellular\\in_sim\\g110_c2k_n01\\subcellular_data_hub"),
        help="Output directory for per-cell transcript CSV files.",
    )
    parser.add_argument("--mode", type=str, default="poisson", choices=["poisson", "round"])
    parser.add_argument("--top-r", type=float, default=None)
    parser.add_argument("--count-scale", type=float, default=20.0)
    parser.add_argument("--r-cell", type=float, default=0.05)
    parser.add_argument("--r-nuc", type=float, default=0.025)
    parser.add_argument("--m-candidates", type=int, default=40000)
    parser.add_argument("--k-hub", type=int, default=3)
    parser.add_argument("--sigma-reg", type=float, default=0.02)
    parser.add_argument("--save-hub-meta", action="store_true")
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument("--limit-cells", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    saved_cells, first_cell = generate_and_save_subcellular_transcripts_hub(
        expr_path=args.expr_path,
        grn_path=args.grn_path,
        output_dir=args.output_dir,
        top_r=args.top_r,
        mode=args.mode,
        count_scale=args.count_scale,
        r_cell=args.r_cell,
        r_nuc=args.r_nuc,
        m_candidates=args.m_candidates,
        k_hub=args.k_hub,
        sigma_reg=args.sigma_reg,
        save_hub_meta=args.save_hub_meta,
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


if __name__ == "__main__":
    main()