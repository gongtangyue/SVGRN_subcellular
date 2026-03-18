import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from eval import get_AUPRC_AUROC, get_evaluation_matrix, get_truth_edges, evaluate


def _to_float_or_none(values):
    if not values:
        return None
    return float(np.mean(values))


def _save_metrics(run_dir, stage_name, metrics):
    run_dir.mkdir(parents=True, exist_ok=True)
    json_path = run_dir / "eval_allcell_metrics.json"
    csv_path = run_dir / "eval_allcell_metrics.csv"

    payload = {"stage": stage_name, **metrics}
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    pd.DataFrame([payload]).to_csv(csv_path, index=False)
    return json_path, csv_path


def _build_truth_edges_map(cell_name_list, gt_scgrn_folder):
    truth_edges_map = {}
    for cell_name in cell_name_list:
        gt_file = gt_scgrn_folder / f"{cell_name}.csv"
        if gt_file.exists():
            truth_edges_map[cell_name] = get_truth_edges(str(gt_file))
    return truth_edges_map


def _summarize_result(tp_list, ep_list, epr_list, auprc_list, auprc_ratio_list, auroc_list, processed):
    return {
        "processed_cells": int(processed),
        "TP": _to_float_or_none(tp_list),
        "EP": _to_float_or_none(ep_list),
        "EPR": _to_float_or_none(epr_list),
        "AUPRC": _to_float_or_none(auprc_list),
        "AUPRC Ratio": _to_float_or_none(auprc_ratio_list),
        "AUROC": _to_float_or_none(auroc_list),
    }


def evaluate_stage2(stage2_run_dir, rn_filename, truth_edges_map, evaluate_mask):
    tp_list, ep_list, epr_list = [], [], []
    auprc_list, auprc_ratio_list, auroc_list = [], [], []
    processed = 0

    for cell_name, truth_edges in truth_edges_map.items():
        csv_path = stage2_run_dir / cell_name / rn_filename
        if not csv_path.exists():
            continue

        pre_grn = pd.read_csv(csv_path).values.T
        tp, ep, epr = evaluate(pre_grn, truth_edges, Evaluate_Mask=evaluate_mask, cutoff_factor=1)
        tp_list.append(tp)
        ep_list.append(ep)
        epr_list.append(epr)

        auprc, auprc_ratio, auroc = get_AUPRC_AUROC(pre_grn, truth_edges, Evaluate_Mask=evaluate_mask)
        if auprc is not None:
            auprc_list.append(auprc)
            auprc_ratio_list.append(auprc_ratio)
            auroc_list.append(auroc)

        processed += 1

    return _summarize_result(
        tp_list, ep_list, epr_list, auprc_list, auprc_ratio_list, auroc_list, processed
    )


def evaluate_stage1(stage1_run_dir, stage1_rn_filename, truth_edges_map, evaluate_mask):
    stage1_csv = stage1_run_dir / stage1_rn_filename
    if not stage1_csv.exists():
        raise FileNotFoundError(f"Stage1 RN file not found: {stage1_csv}")

    stage1_grn = pd.read_csv(stage1_csv).values.T

    tp_list, ep_list, epr_list = [], [], []
    auprc_list, auprc_ratio_list, auroc_list = [], [], []
    processed = 0

    for truth_edges in truth_edges_map.values():
        tp, ep, epr = evaluate(stage1_grn, truth_edges, Evaluate_Mask=evaluate_mask, cutoff_factor=1)
        tp_list.append(tp)
        ep_list.append(ep)
        epr_list.append(epr)

        auprc, auprc_ratio, auroc = get_AUPRC_AUROC(stage1_grn, truth_edges, Evaluate_Mask=evaluate_mask)
        if auprc is not None:
            auprc_list.append(auprc)
            auprc_ratio_list.append(auprc_ratio)
            auroc_list.append(auroc)

        processed += 1

    return _summarize_result(
        tp_list, ep_list, epr_list, auprc_list, auprc_ratio_list, auroc_list, processed
    )


def build_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate stage1/stage2 all-cell GRN results and save summaries in each run directory."
    )
    parser.add_argument("--expr_file", type=Path, required=True, help="Expression CSV used to derive gene list.")
    parser.add_argument("--tf_file", type=Path, required=True, help="TF list text file.")
    parser.add_argument("--cellname_file", type=Path, required=True, help="Cell name list .npy file.")
    parser.add_argument("--gt_scgrn_folder", type=Path, required=True, help="Folder of per-cell ground-truth GRNs.")
    parser.add_argument("--stage1_run_dir", type=Path, required=True, help="Stage1 run directory.")
    parser.add_argument("--stage2_run_dir", type=Path, required=True, help="Stage2 run directory.")
    parser.add_argument("--rn_filename", type=str, default="RN_150.csv", help="RN filename for stage2 per-cell output.")
    parser.add_argument(
        "--stage1_rn_filename",
        type=str,
        default="RN_150.csv",
        help="RN filename under stage1 run directory.",
    )
    return parser


def main():
    args = build_parser().parse_args()

    all_data = pd.read_csv(args.expr_file, index_col=0)
    gene_list = [
        str(g)
        for g in all_data.drop(columns=["x", "y", "ClusterID"], errors="ignore").columns
    ]
    tf_list = [line.strip() for line in args.tf_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    cell_name_list = np.load(args.cellname_file, allow_pickle=True).astype(str).ravel().tolist()

    evaluate_mask = get_evaluation_matrix(tf_list, gene_list)
    truth_edges_map = _build_truth_edges_map(cell_name_list, args.gt_scgrn_folder)
    if not truth_edges_map:
        raise RuntimeError("No valid ground-truth files found for the provided cell list.")

    stage1_metrics = evaluate_stage1(
        args.stage1_run_dir, args.stage1_rn_filename, truth_edges_map, evaluate_mask
    )
    stage2_metrics = evaluate_stage2(
        args.stage2_run_dir, args.rn_filename, truth_edges_map, evaluate_mask
    )

    stage1_json, stage1_csv = _save_metrics(args.stage1_run_dir, "stage1", stage1_metrics)
    stage2_json, stage2_csv = _save_metrics(args.stage2_run_dir, "stage2", stage2_metrics)

    print("----- Stage1 Summary -----")
    for k, v in stage1_metrics.items():
        print(f"{k}: {v}")
    print(f"saved: {stage1_json}")
    print(f"saved: {stage1_csv}")

    print("----- Stage2 Summary -----")
    for k, v in stage2_metrics.items():
        print(f"{k}: {v}")
    print(f"saved: {stage2_json}")
    print(f"saved: {stage2_csv}")


if __name__ == "__main__":
    main()
