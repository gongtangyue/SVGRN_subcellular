import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from eval import evaluate, get_AUPRC_AUROC, get_evaluation_matrix, get_truth_edges
from gen_data_hub import generate_and_save_subcellular_transcripts_hub
from src.subcellular_colocalization import compute_gaussian_colocalization
from src.subcellular_feature_store import min_max_scale


def parse_param(value):
    parts = value.split(":")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Parameter must be formatted as k_hub:sigma_reg")
    return int(parts[0]), float(parts[1])


def truth_edges_to_matrix(truth_edges, n_genes):
    truth = np.zeros((n_genes, n_genes), dtype=bool)
    for receptor, sender in truth_edges:
        truth[sender, receptor] = True
    return truth


def generate_coloc_only(
    expr_file,
    subcellular_data_dir,
    feature_dir,
    subcell_sigma,
    progress_every,
    limit_cells,
):
    expr_df = pd.read_csv(expr_file, index_col=0)
    expr_df.index = expr_df.index.astype(str)
    expr_df.columns = expr_df.columns.astype(str)
    if limit_cells is not None:
        expr_df = expr_df.iloc[: int(limit_cells)]

    gene_names = [col for col in expr_df.columns if col not in ("x", "y", "ClusterID")]
    cell_ids = expr_df.index.astype(str).to_numpy()
    coloc_features = []
    missing_or_empty = 0

    for idx, cell_id in enumerate(cell_ids, start=1):
        csv_path = Path(subcellular_data_dir) / f"{cell_id}_subcellular.csv"
        if csv_path.exists():
            sub_df = pd.read_csv(csv_path)
        else:
            sub_df = pd.DataFrame(columns=["gene", "x", "y"])
            missing_or_empty += 1

        coloc = compute_gaussian_colocalization(sub_df, gene_names, subcell_sigma)
        coloc_features.append(coloc.astype(np.float32, copy=False))

        if progress_every > 0 and (idx % progress_every == 0 or idx == len(cell_ids)):
            print(f"[coloc {idx}/{len(cell_ids)}] processed {cell_id}")

    subcell_coloc = np.stack(coloc_features, axis=0).astype(np.float32, copy=False)
    subcell_coloc = min_max_scale(subcell_coloc.reshape(len(cell_ids), -1))
    subcell_coloc = subcell_coloc.reshape(len(cell_ids), len(gene_names), len(gene_names))

    feature_dir = Path(feature_dir)
    feature_dir.mkdir(parents=True, exist_ok=True)
    np.save(feature_dir / "cell_ids.npy", cell_ids.astype(str))
    np.save(feature_dir / "gene_names.npy", np.asarray(gene_names, dtype=str))
    np.save(feature_dir / "subcell_coloc.npy", subcell_coloc.astype(np.float32, copy=False))
    (feature_dir / "subcell_feature_config.json").write_text(
        json.dumps(
            {
                "coloc_only": True,
                "subcell_sigma": float(subcell_sigma),
                "missing_or_empty": int(missing_or_empty),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return cell_ids, np.asarray(gene_names, dtype=str), subcell_coloc


def evaluate_coloc(feature_dir, gt_dir, tf_file):
    feature_dir = Path(feature_dir)
    gt_dir = Path(gt_dir)
    cell_ids = np.load(feature_dir / "cell_ids.npy", allow_pickle=False).astype(str)
    gene_names = np.load(feature_dir / "gene_names.npy", allow_pickle=False).astype(str)
    coloc = np.load(feature_dir / "subcell_coloc.npy", allow_pickle=False, mmap_mode="r")

    tf_list = [
        line.strip()
        for line in Path(tf_file).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    eval_mask = get_evaluation_matrix(tf_list, gene_names).astype(bool)

    rows = []
    true_scores_all = []
    false_scores_all = []
    for cell_idx, cell_id in enumerate(cell_ids):
        gt_path = gt_dir / f"{cell_id}.csv"
        if not gt_path.exists():
            continue

        prediction = np.asarray(coloc[cell_idx])
        truth_edges = get_truth_edges(str(gt_path))
        auprc, auprc_ratio, auroc = get_AUPRC_AUROC(
            prediction, truth_edges, Evaluate_Mask=eval_mask
        )
        if auprc is None:
            continue

        tp, ep, epr = evaluate(prediction, truth_edges, Evaluate_Mask=eval_mask)
        truth = truth_edges_to_matrix(truth_edges, len(gene_names))
        y_true = truth[eval_mask]
        y_score = prediction[eval_mask]
        true_scores = y_score[y_true == 1]
        false_scores = y_score[y_true == 0]

        rows.append(
            {
                "cell": cell_id,
                "pos_edges": int(y_true.sum()),
                "neg_edges": int(len(y_true) - y_true.sum()),
                "TP": tp,
                "EP": ep,
                "EPR": epr,
                "AUROC": auroc,
                "AUPRC": auprc,
                "AUPRC_ratio": auprc_ratio,
                "true_mean": float(true_scores.mean()),
                "false_mean": float(false_scores.mean()),
                "true_false_mean_ratio": float(true_scores.mean() / false_scores.mean())
                if false_scores.mean() != 0
                else np.inf,
            }
        )
        true_scores_all.append(true_scores)
        false_scores_all.append(false_scores)

    per_cell = pd.DataFrame(rows)
    if per_cell.empty:
        raise RuntimeError(f"No evaluable cells for {feature_dir}")

    pooled_true = np.concatenate(true_scores_all)
    pooled_false = np.concatenate(false_scores_all)
    rng = np.random.default_rng(1)
    true_sample = rng.choice(pooled_true, size=200_000, replace=True)
    false_sample = rng.choice(pooled_false, size=200_000, replace=True)

    summary = per_cell.mean(numeric_only=True).to_dict()
    summary["processed_cells"] = int(len(per_cell))
    summary["pooled_true_mean"] = float(pooled_true.mean())
    summary["pooled_false_mean"] = float(pooled_false.mean())
    summary["pooled_true_false_mean_ratio"] = float(pooled_true.mean() / pooled_false.mean())
    summary["sample_P_true_score_gt_false_score"] = float(np.mean(true_sample > false_sample))
    return per_cell, summary


def main():
    parser = argparse.ArgumentParser(
        description="Generate hub subcellular data for one parameter setting and evaluate coloc.npy."
    )
    parser.add_argument("--expr-file", required=True, type=Path)
    parser.add_argument("--grn-dir", required=True, type=Path)
    parser.add_argument("--tf-file", required=True, type=Path)
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--param", required=True, type=parse_param, help="Format: k_hub:sigma_reg")
    parser.add_argument("--top-r", type=float, default=10.0)
    parser.add_argument("--count-scale", type=float, default=20.0)
    parser.add_argument("--r-cell", type=float, default=0.05)
    parser.add_argument("--r-nuc", type=float, default=0.025)
    parser.add_argument("--m-candidates", type=int, default=40000)
    parser.add_argument("--subcell-sigma", type=float, default=0.03)
    parser.add_argument("--limit-cells", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    k_hub, sigma_reg = args.param
    tag = f"k{k_hub}_sigma{sigma_reg:g}".replace(".", "p")
    out_dir = args.out_root / tag
    data_dir = out_dir / "subcellular_data"
    feature_dir = out_dir / "subcellular_features_coloc"
    out_dir.mkdir(parents=True, exist_ok=True)

    first_cell = pd.read_csv(args.expr_file, index_col=0, nrows=1).index.astype(str)[0]
    first_csv = data_dir / f"{first_cell}_subcellular.csv"
    if args.force or not first_csv.exists():
        generate_and_save_subcellular_transcripts_hub(
            expr_path=args.expr_file,
            grn_path=args.grn_dir,
            output_dir=data_dir,
            top_r=args.top_r,
            mode="poisson",
            count_scale=args.count_scale,
            r_cell=args.r_cell,
            r_nuc=args.r_nuc,
            m_candidates=args.m_candidates,
            k_hub=k_hub,
            sigma_reg=sigma_reg,
            save_hub_meta=True,
            progress_every=args.progress_every,
            limit_cells=args.limit_cells,
        )
    else:
        print(f"Reusing existing subcellular data: {data_dir}")

    coloc_path = feature_dir / "subcell_coloc.npy"
    if args.force or not coloc_path.exists():
        generate_coloc_only(
            expr_file=args.expr_file,
            subcellular_data_dir=data_dir,
            feature_dir=feature_dir,
            subcell_sigma=args.subcell_sigma,
            progress_every=args.progress_every,
            limit_cells=args.limit_cells,
        )
    else:
        print(f"Reusing existing coloc features: {feature_dir}")

    per_cell, summary = evaluate_coloc(feature_dir, args.grn_dir, args.tf_file)
    summary.update(
        {
            "tag": tag,
            "k_hub": int(k_hub),
            "sigma_reg": float(sigma_reg),
            "top_r": float(args.top_r),
            "data_dir": str(data_dir),
            "feature_dir": str(feature_dir),
        }
    )
    per_cell.to_csv(out_dir / "coloc_eval_per_cell.csv", index=False)
    pd.DataFrame([summary]).to_csv(out_dir / "coloc_eval_summary.csv", index=False)
    print(pd.DataFrame([summary]).to_string(index=False))


if __name__ == "__main__":
    main()
