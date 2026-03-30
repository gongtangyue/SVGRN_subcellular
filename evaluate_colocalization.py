import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from eval import get_AUPRC_AUROC, get_evaluation_matrix, get_truth_edges, evaluate


def _mean_or_none(values):
    if not values:
        return None
    return float(np.mean(values))


def _load_tf_list(tf_file):
    if tf_file is None:
        return None
    return [line.strip() for line in tf_file.read_text(encoding="utf-8").splitlines() if line.strip()]


def _build_directed_mask(tf_list, gene_list):
    if tf_list is None:
        mask = np.ones((len(gene_list), len(gene_list)), dtype=int)
        np.fill_diagonal(mask, 0)
        return mask
    return get_evaluation_matrix(tf_list, gene_list)


def _build_undirected_mask(tf_list, gene_list):
    mask = np.triu(np.ones((len(gene_list), len(gene_list)), dtype=bool), k=1)
    if tf_list is None:
        return mask

    tf_set = set(tf_list)
    is_tf = np.array([gene in tf_set for gene in gene_list], dtype=bool)
    tf_pair_mask = np.logical_or.outer(is_tf, is_tf)
    return mask & tf_pair_mask


def _to_undirected_truth_pairs(truth_edges):
    truth_pairs = set()
    for receptor, sender in truth_edges:
        if receptor == sender:
            continue
        pair = (min(receptor, sender), max(receptor, sender))
        truth_pairs.add(pair)
    return truth_pairs


def _pick_threshold(sorted_scores_desc, num_truth_edges, cutoff_factor=1):
    if num_truth_edges <= 0 or sorted_scores_desc.size == 0:
        return None

    cutoff_idx = min(num_truth_edges * cutoff_factor, sorted_scores_desc.size) - 1
    cutoff = float(sorted_scores_desc[cutoff_idx])

    if cutoff == 0 and np.all(sorted_scores_desc == 0):
        return 0.1
    if cutoff == 0:
        nonzero = sorted_scores_desc[sorted_scores_desc != 0]
        if nonzero.size > 0:
            return float(nonzero.min())
    return cutoff


def evaluate_undirected(score_matrix, truth_pairs, evaluate_mask, cutoff_factor=1):
    masked_scores = score_matrix[evaluate_mask]
    num_truth_pairs = len(truth_pairs)
    num_eval_pairs = int(evaluate_mask.sum())

    if num_truth_pairs == 0 or num_eval_pairs == 0:
        return None

    sorted_scores = np.sort(np.abs(masked_scores).ravel())[::-1]
    cutoff = _pick_threshold(sorted_scores, num_truth_pairs, cutoff_factor=cutoff_factor)
    if cutoff is None:
        return None

    pred_mask = evaluate_mask & (np.abs(score_matrix) >= cutoff)
    pred_pairs = set(zip(*np.where(pred_mask)))
    overlap = pred_pairs.intersection(truth_pairs)
    tp = len(overlap)
    ep = tp / num_truth_pairs
    epr = tp / ((num_truth_pairs ** 2) / num_eval_pairs)
    return tp, ep, epr


def get_undirected_auprc_auroc(score_matrix, truth_pairs, evaluate_mask):
    num_eval_pairs = int(evaluate_mask.sum())
    if num_eval_pairs == 0:
        return None, None, None

    truth_matrix = np.zeros_like(score_matrix, dtype=np.int8)
    for i, j in truth_pairs:
        truth_matrix[i, j] = 1

    y_true = truth_matrix[evaluate_mask].ravel()
    y_score = score_matrix[evaluate_mask].ravel()

    pos = int(y_true.sum())
    neg = int(len(y_true) - pos)
    if pos == 0 or neg == 0:
        return None, None, None

    auprc = average_precision_score(y_true, y_score)
    prevalence = y_true.mean()
    auprc_ratio = auprc / prevalence
    auroc = roc_auc_score(y_true, y_score)
    return auprc, auprc_ratio, auroc


def summarize_metrics(records, prefix):
    return {
        f"{prefix}_processed_cells": int(sum(record.get(f"{prefix}_processed", False) for record in records)),
        f"{prefix}_TP": _mean_or_none([record[f"{prefix}_TP"] for record in records if record[f"{prefix}_processed"]]),
        f"{prefix}_EP": _mean_or_none([record[f"{prefix}_EP"] for record in records if record[f"{prefix}_processed"]]),
        f"{prefix}_EPR": _mean_or_none([record[f"{prefix}_EPR"] for record in records if record[f"{prefix}_processed"]]),
        f"{prefix}_AUPRC": _mean_or_none(
            [record[f"{prefix}_AUPRC"] for record in records if record[f"{prefix}_processed"] and record[f"{prefix}_AUPRC"] is not None]
        ),
        f"{prefix}_AUPRC_ratio": _mean_or_none(
            [
                record[f"{prefix}_AUPRC_ratio"]
                for record in records
                if record[f"{prefix}_processed"] and record[f"{prefix}_AUPRC_ratio"] is not None
            ]
        ),
        f"{prefix}_AUROC": _mean_or_none(
            [record[f"{prefix}_AUROC"] for record in records if record[f"{prefix}_processed"] and record[f"{prefix}_AUROC"] is not None]
        ),
        f"{prefix}_mean_truth_edges": _mean_or_none(
            [record[f"{prefix}_truth_edges"] for record in records if record[f"{prefix}_processed"]]
        ),
    }


def evaluate_coloc_per_cell(coloc_array, cell_ids, gene_list, gt_scgrn_folder, directed_mask, undirected_mask, mode):
    records = []

    for idx, cell_id in enumerate(cell_ids):
        gt_file = gt_scgrn_folder / f"{cell_id}.csv"
        record = {"cell_id": cell_id}
        if not gt_file.exists():
            record["gt_found"] = False
            for prefix in ("directed", "undirected"):
                record[f"{prefix}_processed"] = False
                record[f"{prefix}_truth_edges"] = None
                record[f"{prefix}_TP"] = None
                record[f"{prefix}_EP"] = None
                record[f"{prefix}_EPR"] = None
                record[f"{prefix}_AUPRC"] = None
                record[f"{prefix}_AUPRC_ratio"] = None
                record[f"{prefix}_AUROC"] = None
            records.append(record)
            continue

        score_matrix = np.asarray(coloc_array[idx], dtype=np.float32)
        if score_matrix.shape != (len(gene_list), len(gene_list)):
            raise ValueError(
                f"Cell {cell_id} colocalization shape mismatch: expected {(len(gene_list), len(gene_list))}, got {score_matrix.shape}"
            )
        score_matrix = score_matrix.copy()
        np.fill_diagonal(score_matrix, 0.0)

        truth_edges = get_truth_edges(str(gt_file))
        truth_pairs = _to_undirected_truth_pairs(truth_edges)

        record["gt_found"] = True

        if mode in {"directed", "both"}:
            tp, ep, epr = evaluate(score_matrix, truth_edges, Evaluate_Mask=directed_mask, cutoff_factor=1)
            auprc, auprc_ratio, auroc = get_AUPRC_AUROC(score_matrix, truth_edges, Evaluate_Mask=directed_mask)
            record["directed_processed"] = True
            record["directed_truth_edges"] = len(truth_edges)
            record["directed_TP"] = tp
            record["directed_EP"] = ep
            record["directed_EPR"] = epr
            record["directed_AUPRC"] = auprc
            record["directed_AUPRC_ratio"] = auprc_ratio
            record["directed_AUROC"] = auroc
        else:
            record["directed_processed"] = False
            record["directed_truth_edges"] = None
            record["directed_TP"] = None
            record["directed_EP"] = None
            record["directed_EPR"] = None
            record["directed_AUPRC"] = None
            record["directed_AUPRC_ratio"] = None
            record["directed_AUROC"] = None

        if mode in {"undirected", "both"}:
            metric_tuple = evaluate_undirected(score_matrix, truth_pairs, evaluate_mask=undirected_mask, cutoff_factor=1)
            if metric_tuple is None:
                tp, ep, epr = None, None, None
            else:
                tp, ep, epr = metric_tuple
            auprc, auprc_ratio, auroc = get_undirected_auprc_auroc(score_matrix, truth_pairs, evaluate_mask=undirected_mask)
            record["undirected_processed"] = True
            record["undirected_truth_edges"] = len(truth_pairs)
            record["undirected_TP"] = tp
            record["undirected_EP"] = ep
            record["undirected_EPR"] = epr
            record["undirected_AUPRC"] = auprc
            record["undirected_AUPRC_ratio"] = auprc_ratio
            record["undirected_AUROC"] = auroc
        else:
            record["undirected_processed"] = False
            record["undirected_truth_edges"] = None
            record["undirected_TP"] = None
            record["undirected_EP"] = None
            record["undirected_EPR"] = None
            record["undirected_AUPRC"] = None
            record["undirected_AUPRC_ratio"] = None
            record["undirected_AUROC"] = None

        records.append(record)

    return records


def build_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate per-cell subcellular colocalization matrices against per-cell ground-truth GRNs."
    )
    parser.add_argument(
        "--feature_dir",
        type=Path,
        required=True,
        help="Directory containing subcell_coloc.npy, cell_ids.npy, and gene_names.npy.",
    )
    parser.add_argument(
        "--gt_scgrn_folder",
        type=Path,
        required=True,
        help="Directory containing per-cell ground-truth GRN csv files.",
    )
    parser.add_argument(
        "--tf_file",
        type=Path,
        default=None,
        help="Optional TF list. If provided, evaluation is restricted to TF-related pairs.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=("directed", "undirected", "both"),
        default="both",
        help="Evaluation mode for colocalization scores.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <feature_dir>/coloc_eval.",
    )
    return parser


def main():
    args = build_parser().parse_args()

    feature_dir = args.feature_dir.expanduser().resolve()
    gt_scgrn_folder = args.gt_scgrn_folder.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else feature_dir / "coloc_eval"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    coloc_array = np.load(feature_dir / "subcell_coloc.npy", allow_pickle=False)
    cell_ids = np.load(feature_dir / "cell_ids.npy", allow_pickle=True).astype(str).ravel().tolist()
    gene_list = np.load(feature_dir / "gene_names.npy", allow_pickle=True).astype(str).ravel().tolist()

    if coloc_array.shape[0] != len(cell_ids):
        raise ValueError(
            f"Cell dimension mismatch: subcell_coloc has {coloc_array.shape[0]} cells, cell_ids has {len(cell_ids)}"
        )
    if coloc_array.shape[1:] != (len(gene_list), len(gene_list)):
        raise ValueError(
            "Gene dimension mismatch: "
            f"subcell_coloc has {coloc_array.shape[1:]}, expected {(len(gene_list), len(gene_list))}"
        )

    tf_list = _load_tf_list(args.tf_file)
    directed_mask = _build_directed_mask(tf_list, gene_list)
    undirected_mask = _build_undirected_mask(tf_list, gene_list)

    records = evaluate_coloc_per_cell(
        coloc_array=coloc_array,
        cell_ids=cell_ids,
        gene_list=gene_list,
        gt_scgrn_folder=gt_scgrn_folder,
        directed_mask=directed_mask,
        undirected_mask=undirected_mask,
        mode=args.mode,
    )

    summary = {
        "feature_dir": str(feature_dir),
        "gt_scgrn_folder": str(gt_scgrn_folder),
        "mode": args.mode,
        "total_cells": len(cell_ids),
        "gt_found_cells": int(sum(record["gt_found"] for record in records)),
    }
    if args.mode in {"directed", "both"}:
        summary.update(summarize_metrics(records, "directed"))
    if args.mode in {"undirected", "both"}:
        summary.update(summarize_metrics(records, "undirected"))

    per_cell_csv = output_dir / "coloc_eval_per_cell.csv"
    summary_csv = output_dir / "coloc_eval_summary.csv"
    summary_json = output_dir / "coloc_eval_summary.json"

    pd.DataFrame(records).to_csv(per_cell_csv, index=False)
    pd.DataFrame([summary]).to_csv(summary_csv, index=False)
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("----- Colocalization Evaluation Summary -----")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"saved: {per_cell_csv}")
    print(f"saved: {summary_csv}")
    print(f"saved: {summary_json}")


if __name__ == "__main__":
    main()
