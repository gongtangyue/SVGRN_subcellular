import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


def natural_key(text: str):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def build_tf_list(dataset_dir: Path) -> list[str]:
    allcells_grn = dataset_dir / "allcells_GRN.csv"
    if not allcells_grn.exists():
        raise FileNotFoundError(f"Missing file: {allcells_grn}")

    df = pd.read_csv(allcells_grn, index_col=0)
    tf_list = [str(col).strip() for col in df.columns if str(col).strip()]
    if not tf_list:
        raise ValueError(f"No TF columns found in: {allcells_grn}")
    return tf_list


def build_cellname_list(dataset_dir: Path) -> list[str]:
    cell_grn_dir = dataset_dir / "cell_specific_GRN"
    if not cell_grn_dir.exists():
        raise FileNotFoundError(f"Missing directory: {cell_grn_dir}")

    cell_names = [p.stem for p in cell_grn_dir.glob("*.csv")]
    cell_names = sorted(cell_names, key=natural_key)
    if not cell_names:
        raise ValueError(f"No cell GRN csv files found in: {cell_grn_dir}")
    return cell_names


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create TF_list.txt and cellname_list_all.npy in a dataset folder."
    )
    parser.add_argument(
        "dataset_folder",
        help="Dataset folder path, e.g. in_sim/g110_c2k_n05 or E:/st/SVGRN/in_sim/g110_c2k_n05",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_dir = Path(args.dataset_folder).resolve()
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_dir}")
    if not dataset_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {dataset_dir}")

    tf_list = build_tf_list(dataset_dir)
    cell_names = build_cellname_list(dataset_dir)

    tf_out = dataset_dir / "TF_list.txt"
    cell_out = dataset_dir / "cellname_list_all.npy"

    tf_out.write_text("\n".join(tf_list), encoding="utf-8")
    np.save(cell_out, np.array(cell_names, dtype=str))

    print(f"Dataset: {dataset_dir}")
    print(f"Created: {tf_out} (TF count={len(tf_list)})")
    print(f"Created: {cell_out} (cell count={len(cell_names)})")


if __name__ == "__main__":
    main()
