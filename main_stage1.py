import argparse
import json
import multiprocessing
import os
from datetime import datetime

from src.SVGRN_allcell import non_celltype_GRN_model

#python main_stage1.py `
#  --task simulation_allcell_GRN `
#  --setting default `
#  --data_file "E:/st/SVGRN_subcellular/in_sim/g110_c2k_n01/expression_loc_cluster_wlayout.csv" `
#  --tf_list "E:/st/SVGRN_subcellular/in_sim/g110_c2k_n01/TF_list.txt" `
#  --net_file "E:/st/SVGRN_subcellular/in_sim/g110_c2k_n01/allcells_GRN.csv" `
#  --subcellular_data_dir "E:/st/SVGRN_subcellular/in_sim/g110_c2k_n01/subcellular_data" `
#  --save_name "E:/st/SVGRN_subcellular/runs/stage1_subcellular"


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=120, help="Number of epochs.")
    parser.add_argument(
        "--task",
        type=str,
        default="celltype_GRN",
        help="Determine which task to run.",
    )
    parser.add_argument("--setting", type=str, default="default", help="Use default hyper-parameters or not.")
    parser.add_argument("--batch_size", type=int, default=64, help="Training batch size.")
    parser.add_argument("--data_file", type=str, help="Input gene expression file.")
    parser.add_argument("--net_file", type=str, default=None, help="Optional ground-truth GRN file.")
    parser.add_argument("--tf_list", type=str, help="TF list file.")
    parser.add_argument(
        "--subcellular_data_dir",
        type=str,
        default="",
        help="Directory containing per-cell subcellular CSV files.",
    )
    parser.add_argument(
        "--subcellular_feature_dir",
        type=str,
        default="",
        help="Directory containing precomputed subcellular feature files.",
    )
    parser.add_argument(
        "--subcell_sigma",
        type=float,
        default=0.03,
        help="Gaussian sigma used for transcript colocalization weights.",
    )
    parser.add_argument(
        "--subcell_grid_size",
        type=int,
        default=16,
        help="Grid size for subcellular transcript splatting.",
    )
    parser.add_argument(
        "--subcell_r_cell",
        type=float,
        default=0.05,
        help="Cell radius used to normalize transcript coordinates into local grid space.",
    )
    parser.add_argument(
        "--subcell_splat_sigma",
        type=float,
        default=1.0,
        help="Gaussian sigma for transcript splatting in grid-cell units.",
    )
    parser.add_argument(
        "--subcell_splat_radius",
        type=int,
        default=1,
        help="Neighborhood radius for Gaussian splatting.",
    )
    parser.add_argument(
        "--subcell_grid_norm",
        type=str,
        default="log1p",
        help="Normalization mode for the gene grid: log1p, per_gene_sum, or none.",
    )
    parser.add_argument(
        "--subcell_gene_pool_channels",
        type=int,
        default=16,
        help="Output channels of the 1x1 gene-pooling convolution.",
    )
    parser.add_argument(
        "--subcell_cnn_hidden",
        type=int,
        default=32,
        help="Hidden channels used in the spatial CNN over the subcellular grid.",
    )
    parser.add_argument(
        "--y_prime_coloc_dim",
        type=int,
        default=64,
        help="Output embedding dimension for the colocalization encoder.",
    )
    parser.add_argument(
        "--y_prime_grid_dim",
        type=int,
        default=64,
        help="Output embedding dimension for the subcellular grid encoder.",
    )
    parser.add_argument("--alpha", type=float, default=100, help="L1 coefficient for W.")
    parser.add_argument("--beta", type=float, default=1, help="KL loss coefficient.")
    parser.add_argument("--lr", type=float, default=1e-4, help="RMSprop learning rate.")
    parser.add_argument("--lr_step_size", type=int, default=1, help="Step size for LR scheduler.")
    parser.add_argument("--gamma", type=float, default=0.95, help="LR decay factor.")
    parser.add_argument("--n_hidden", type=int, default=128, help="Number of hidden units in MLP.")
    parser.add_argument("--K", type=int, default=1, help="Number of Gaussian kernels in GMM.")
    parser.add_argument("--K1", type=int, default=1, help="Epoch count for optimizing MLP.")
    parser.add_argument("--K2", type=int, default=2, help="Epoch count for optimizing W.")
    parser.add_argument("--save_name", type=str, default="/tmp")

    # Stage-2 params kept for compatibility with existing command templates.
    parser.add_argument("--model_file", type=str, default="", help="Loaded stage-1 model path.")
    parser.add_argument("--target_cell_name", type=str, default="", help="Target cell name for stage-2.")

    parser.add_argument("--GPU", action="store_true", help="Use GPU or not.")
    parser.add_argument("--device", type=str, default="", help="cpu or gpu")
    parser.add_argument("--dropout_mask", action="store_true", help="Use dropout mask or not.")
    return parser


def run():
    parser = build_parser()
    opt = parser.parse_args()

    if opt.task != "simulation_allcell_GRN":
        raise ValueError("Unknown task. Please use simulation_allcell_GRN task.")

    if opt.setting == "default":
        opt.n_epochs = 150
        opt.K1 = 1
        opt.K2 = 2
        opt.n_hidden = 128
        opt.gamma = 0.95
        opt.lr = 1e-4
        opt.lr_step_size = 0.99
        opt.batch_size = 64

    opt.subcellular_feature_dir = opt.subcellular_feature_dir or os.path.join(
        os.path.dirname(os.path.abspath(opt.data_file)),
        "subcellular_features",
    )

    # Create a timestamped run directory for this training job.
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    opt.save_name = os.path.join(opt.save_name, run_tag)

    model = non_celltype_GRN_model(opt)
    with open(os.path.join(opt.save_name, "args.txt"), "a") as f:
        json.dump(opt.__dict__, f, indent=2)
    print(f"Run outputs will be saved to: {opt.save_name}")
    model.train_model()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    run()
