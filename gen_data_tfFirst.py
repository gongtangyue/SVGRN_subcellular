import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from gen_data import (
    sample_counts,
    sample_points_by_lambda,
    sample_uniform_in_disk,
    softplus,
    u_gene_bias,
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
                reg_weights.append((str(reg), weight))
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


def build_tf_target_sets(target_to_regs):
    tf_targets = {}
    tf_target_weights = {}
    for target, regs in target_to_regs.items():
        for reg, weight in regs:
            tf_targets.setdefault(reg, set()).add(target)
            tf_target_weights.setdefault(reg, {})[target] = float(weight)
    return tf_targets, tf_target_weights


def shared_target_score(tf_a, tf_b, tf_target_weights):
    a = tf_target_weights.get(tf_a, {})
    b = tf_target_weights.get(tf_b, {})
    shared = set(a).intersection(b)
    if not shared:
        return 0.0
    return sum(min(a[t], b[t]) for t in shared)


def assign_tfs_to_hubs(target_to_regs, counts_by_gene, row, hubs, coop_strength=2.0):
    tf_targets, tf_target_weights = build_tf_target_sets(target_to_regs)
    regs = sorted(tf_targets)
    if not regs:
        return {}

    degrees = {tf: len(tf_targets[tf]) for tf in regs}
    expr = {tf: max(float(row.get(tf, 0.0)), 0.0) for tf in regs}
    order = sorted(regs, key=lambda tf: (degrees[tf], expr[tf], counts_by_gene.get(tf, 0)), reverse=True)

    assignments = {}
    hub_members = {i: [] for i in range(len(hubs))}

    for seed_idx, tf in enumerate(order[: len(hubs)]):
        hub_idx = seed_idx % len(hubs)
        assignments[tf] = hub_idx
        hub_members[hub_idx].append(tf)

    for tf in order[len(hubs) :]:
        scores = []
        for hub_idx, hub in hubs.iterrows():
            coop = sum(shared_target_score(tf, other, tf_target_weights) for other in hub_members[hub_idx])
            hub_prior = float(hub["strength"])
            scores.append(hub_prior + coop_strength * coop)

        scores = np.asarray(scores, dtype=float)
        if scores.sum() <= 0:
            hub_idx = int(rng.integers(0, len(hubs)))
        else:
            probs = scores / scores.sum()
            hub_idx = int(rng.choice(len(hubs), p=probs))
        assignments[tf] = hub_idx
        hub_members[hub_idx].append(tf)

    return assignments


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


def sample_gene_mixture(
    cand,
    n,
    regulatory_lam,
    basal_lam,
    p_reg=0.7,
    p_basal=0.2,
    p_noise=0.1,
    top_r=None,
):
    weights = np.asarray([p_reg, p_basal, p_noise], dtype=float)
    weights = weights / weights.sum()
    n_reg, n_basal, n_noise = rng.multinomial(n, weights)

    parts = []
    if n_reg > 0:
        parts.append(sample_points_by_lambda(cand, regulatory_lam, n_reg, top_r=top_r))
    if n_basal > 0:
        parts.append(sample_points_by_lambda(cand, basal_lam, n_basal, top_r=top_r))
    if n_noise > 0:
        idx = rng.integers(0, cand.shape[0], size=n_noise)
        parts.append(cand[idx])

    pts = np.vstack(parts) if parts else np.empty((0, 2))
    rng.shuffle(pts)
    return pts


def regulatory_hub_weights(gene, target_to_regs, tf_to_hub, hubs):
    pos_weights = np.zeros(len(hubs), dtype=float)
    neg_weights = np.zeros(len(hubs), dtype=float)
    for reg, edge_weight in target_to_regs.get(str(gene), []):
        hub_idx = tf_to_hub.get(reg)
        if hub_idx is not None:
            edge_weight = float(edge_weight)
            if edge_weight > 0:
                pos_weights[hub_idx] += edge_weight
            elif edge_weight < 0:
                neg_weights[hub_idx] += abs(edge_weight)
    return pos_weights, neg_weights


def signed_regulatory_field(cand, hubs, pos_weights, neg_weights, negative_avoidance=1.0, bias=0.0):
    pos_field = hub_field(cand, hubs, pos_weights, bias=0.0)
    neg_field = hub_field(cand, hubs, neg_weights, bias=0.0)
    if neg_field.max() > 0:
        avoid_field = neg_field.max() - neg_field
    else:
        avoid_field = np.zeros(cand.shape[0], dtype=float)
    return bias + pos_field + negative_avoidance * avoid_field


def summarize_tf_hub_effects(target_to_regs, tf_to_hub, hubs):
    tf_hub_effects = {}
    for regs in target_to_regs.values():
        for reg, edge_weight in regs:
            hub_idx = tf_to_hub.get(reg)
            if hub_idx is None:
                continue
            key = (reg, hub_idx)
            pos_weight, neg_weight = tf_hub_effects.get(key, (0.0, 0.0))
            edge_weight = float(edge_weight)
            if edge_weight > 0:
                pos_weight += edge_weight
            elif edge_weight < 0:
                neg_weight += abs(edge_weight)
            tf_hub_effects[key] = (pos_weight, neg_weight)
    return [
        {
            "TF": tf,
            "hub": f"hub{hub_idx + 1}",
            "positive_weight": pos_weight,
            "negative_weight": -neg_weight,
        }
        for (tf, hub_idx), (pos_weight, neg_weight) in tf_hub_effects.items()
    ]


def generate_one_cell_transcripts_hub(
    cell_id,
    row,
    gene_cols,
    grn_path,
    C=1.0,
    top_r=None,
    mode="poisson",
    count_scale=20.0,
    r_cell=0.05,
    r_nuc=0.025,
    m_candidates=40000,
    k_hub=3,
    sigma_reg=0.02,
    b_reg=0.01,
    beta0=-1.0,
    p_reg=0.7,
    p_basal=0.2,
    p_noise=0.1,
    coop_strength=2.0,
    negative_avoidance=1.0,
):
    expr_values = row[gene_cols].to_numpy(dtype=float)
    counts = sample_counts(expr_values, mode=mode, count_scale=count_scale)
    counts_by_gene = {str(gene): int(n) for gene, n in zip(gene_cols, counts)}

    center = row[["x", "y"]].to_numpy(dtype=float)
    target_to_regs = load_target_regulator_weights(str(cell_id), grn_path, gene_cols)

    hubs = build_hubs(center, r_nuc, K=k_hub, sigma_reg=sigma_reg)
    tf_to_hub = assign_tfs_to_hubs(
        target_to_regs=target_to_regs,
        counts_by_gene=counts_by_gene,
        row=row,
        hubs=hubs,
        coop_strength=coop_strength,
    )

    cand = sample_uniform_in_disk(center, r_cell, m_candidates)
    u = u_gene_bias(cand, center, r_nuc, u_nuc=0.5, u_cyt=0.0)
    basal_lam = softplus(beta0 + u)

    records = []
    for gene, n in zip(gene_cols, counts):
        n = int(n)
        if n <= 0:
            continue

        pos_weights, neg_weights = regulatory_hub_weights(gene, target_to_regs, tf_to_hub, hubs)
        if pos_weights.sum() <= 0 and neg_weights.sum() <= 0 and str(gene) in tf_to_hub:
            pos_weights[tf_to_hub[str(gene)]] = 1.0

        grn_field = signed_regulatory_field(
            cand,
            hubs,
            pos_weights,
            neg_weights,
            negative_avoidance=negative_avoidance,
            bias=b_reg,
        )
        regulatory_lam = softplus(beta0 + C * grn_field + u)
        pts = sample_gene_mixture(
            cand=cand,
            n=n,
            regulatory_lam=regulatory_lam,
            basal_lam=basal_lam,
            p_reg=p_reg,
            p_basal=p_basal,
            p_noise=p_noise,
            top_r=top_r,
        )
        records.append(pd.DataFrame({"gene": str(gene), "x": pts[:, 0], "y": pts[:, 1]}))

    if records:
        return pd.concat(records, ignore_index=True), hubs, tf_to_hub, target_to_regs
    return pd.DataFrame(columns=["gene", "x", "y"]), hubs, tf_to_hub, target_to_regs


def generate_and_save_subcellular_transcripts_hub(
    expr_path,
    grn_path,
    output_dir,
    C=1.0,
    top_r=None,
    mode="poisson",
    count_scale=20.0,
    r_cell=0.05,
    r_nuc=0.025,
    m_candidates=40000,
    k_hub=3,
    sigma_reg=0.02,
    b_reg=0.01,
    beta0=-1.0,
    p_reg=0.7,
    p_basal=0.2,
    p_noise=0.1,
    coop_strength=2.0,
    negative_avoidance=1.0,
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

        df, hubs, tf_to_hub, target_to_regs = generate_one_cell_transcripts_hub(
            cell_id=cell_id,
            row=row,
            gene_cols=gene_cols,
            grn_path=grn_path,
            C=C,
            top_r=top_r,
            mode=mode,
            count_scale=count_scale,
            r_cell=r_cell,
            r_nuc=r_nuc,
            m_candidates=m_candidates,
            k_hub=k_hub,
            sigma_reg=sigma_reg,
            b_reg=b_reg,
            beta0=beta0,
            p_reg=p_reg,
            p_basal=p_basal,
            p_noise=p_noise,
            coop_strength=coop_strength,
            negative_avoidance=negative_avoidance,
        )
        df.to_csv(output_dir / f"{cell_id}_subcellular.csv", index=False)

        if save_hub_meta:
            hubs.to_csv(hub_dir / f"{cell_id}_hubs.csv", index=False)
            pd.DataFrame(
                [{"TF": tf, "hub": f"hub{hub_idx + 1}"} for tf, hub_idx in tf_to_hub.items()]
            ).to_csv(hub_dir / f"{cell_id}_tf_hubs.csv", index=False)
            pd.DataFrame(summarize_tf_hub_effects(target_to_regs, tf_to_hub, hubs)).to_csv(
                hub_dir / f"{cell_id}_tf_hub_effects.csv",
                index=False,
            )

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
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--top-r", type=float, default=None)
    parser.add_argument("--count-scale", type=float, default=20.0)
    parser.add_argument("--r-cell", type=float, default=0.05)
    parser.add_argument("--r-nuc", type=float, default=0.025)
    parser.add_argument("--m-candidates", type=int, default=40000)
    parser.add_argument("--k-hub", type=int, default=3)
    parser.add_argument("--sigma-reg", type=float, default=0.02)
    parser.add_argument("--b-reg", type=float, default=0.01)
    parser.add_argument("--beta0", type=float, default=-1.0)
    parser.add_argument("--p-reg", type=float, default=0.7)
    parser.add_argument("--p-basal", type=float, default=0.2)
    parser.add_argument("--p-noise", type=float, default=0.1)
    parser.add_argument("--coop-strength", type=float, default=2.0)
    parser.add_argument(
        "--negative-avoidance",
        type=float,
        default=1.0,
        help="Strength for pushing target transcripts away from negatively regulating TF hubs.",
    )
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
        C=args.C,
        top_r=args.top_r,
        mode=args.mode,
        count_scale=args.count_scale,
        r_cell=args.r_cell,
        r_nuc=args.r_nuc,
        m_candidates=args.m_candidates,
        k_hub=args.k_hub,
        sigma_reg=args.sigma_reg,
        b_reg=args.b_reg,
        beta0=args.beta0,
        p_reg=args.p_reg,
        p_basal=args.p_basal,
        p_noise=args.p_noise,
        coop_strength=args.coop_strength,
        negative_avoidance=args.negative_avoidance,
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
