import argparse
import os
import numpy as np
import sys
import json
import time
import multiprocessing
from datetime import datetime

from src.SVGRN_singlecell import SC_GRN_model


#python .\main_stage2_allcell.py `
#  --setting default `
#  --data_file .\in_sim\example_data\expression_loc_cluster_wlayout.csv `
#  --cellname_list .\in_sim\example_data\cellname_list_10.npy `
#  --model_file .\runs\stage1_subcellular\20260218_XXXXXX\stage1.pt `
#  --net_path .\in_sim\example_data\cell_specific_GRN `
#  --save_path .\runs\stage2_subcellular



def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=120, help='Number of Epochs for training DeepSEM')
    parser.add_argument('--setting', type=str, default='default', help='Determine whether or not to use the default hyper-parameter')
    parser.add_argument('--batch_size', type=int, default=64, help='The batch size used in the training process.')
    parser.add_argument('--alpha', type=float, default=100, help='The loss coefficient for L1 norm of W, which is same as \\alpha used in our paper.')
    parser.add_argument('--beta', type=float, default=1, help='The loss coefficient for KL term (beta-VAE), which is same as \\beta used in our paper.')
    parser.add_argument('--lr', type=float, default=1e-4, help='The learning rate of used for RMSprop.')
    parser.add_argument('--lr_step_size', type=int, default=0.99, help='The step size of learning rate decay.')
    parser.add_argument('--gamma', type=float, default=0.95, help='The decay factor of learning rate')
    parser.add_argument('--n_hidden', type=int, default=128, help='The Number of hidden neural used in MLP')
    parser.add_argument('--K', type=int, default=1, help='Number of Gaussian kernel in GMM, default =1')
    parser.add_argument('--K1', type=int, default=1, help='The Number of epoch for optimize MLP. Notes that we optimize MLP and W alternately. The default setting denotes to optimize MLP for one epoch then optimize W for two epochs.')
    parser.add_argument('--K2', type=int, default=2, help='The Number of epoch for optimize W. Notes that we optimize MLP and W alternately. The default setting denotes to optimize MLP for one epoch then optimize W for two epochs.')
    parser.add_argument('--W', type=int, default=5, help='weight for loss')

    ############## in and out file path/name ###############
    parser.add_argument('--save_path', type=str, default='/tmp', help='path to store all the single cell GRN')
    parser.add_argument('--save_name', type=str, default='', help='folder in save path to save single cell GRN')
    parser.add_argument('--data_file', type=str, help='The input scRNA-seq gene expression file.')
    parser.add_argument(
        '--subcellular_data_dir',
        type=str,
        default='',
        help='Directory containing per-cell subcellular CSV files.',
    )
    parser.add_argument(
        '--subcellular_feature_dir',
        type=str,
        default='',
        help='Directory containing precomputed subcellular feature files.',
    )
    parser.add_argument(
        '--subcell_gene_pool_channels',
        type=int,
        default=16,
        help='Output channels of the 1x1 gene-pooling convolution.',
    )
    parser.add_argument(
        '--subcell_cnn_hidden',
        type=int,
        default=32,
        help='Hidden channels used in the spatial CNN over the subcellular grid.',
    )
    parser.add_argument(
        '--subcell_encoder_variant',
        type=str,
        choices=('original', 'direct'),
        default='original',
        help='Logged compatibility option; stage2 uses the encoder stored in the loaded stage1 model.',
    )
    parser.add_argument(
        '--y_prime_coloc_dim',
        type=int,
        default=64,
        help='Output embedding dimension for the gene-specific colocalization encoder.',
    )
    parser.add_argument(
        '--y_prime_grid_dim',
        type=int,
        default=64,
        help='Output embedding dimension for the subcellular grid encoder.',
    )
    parser.add_argument('--net_file', type=str, default='',
                        help='The ground truth of GRN. Only used in GRN inference task if available. ')
    parser.add_argument('--net_path', type=str, default=None,
                        help='The folder path of the GT GRN.')

    ####### params for single cell training (stage 2) ########
    parser.add_argument('--model_file', type=str, default='', help='The loaded stage 1 model path')
    parser.add_argument('--target_cell_name', type=str, default='', help='The target cell name for GRN in stage 2')

    ####### if use GPU ################
    parser.add_argument('--GPU', action='store_true', help='Use GPU or not')
    parser.add_argument('--device', type=str, default='', help='cpu or gpu')

    ###### load cell name .npy #######
    parser.add_argument('--cellname_list', type=str, default='', help='cell name ')

    ###### resume training #######
    parser.add_argument('--start_idx', type=int, default=1, help='1-based start index in cellname_list.')
    parser.add_argument('--end_idx', type=int, default=-1, help='1-based end index in cellname_list; <=0 means to the end.')
    parser.add_argument('--resume_run_dir', type=str, default='',
                        help='If set, continue in an existing stage2 run directory instead of creating a new timestamped one.')

    ######### if need dropout mask for sparse data #############
    parser.add_argument('--dropout_mask', action='store_true', help='if need dropout mask')
    return parser


def run():
    parser = build_parser()
    opt = parser.parse_args()

    try:
        os.mkdir(os.path.dirname(opt.save_path))
    except:
        print(f'{os.path.dirname(opt.save_path)} exist')

    try:
        os.mkdir(opt.save_path)
    except:
        print(f'{opt.save_path} exist')

    if opt.setting == 'default':
        opt.n_epochs = 150   #120 150
        opt.n_hidden = 128
        opt.gamma = 0.95
        opt.lr_step_size = 0.99
        opt.batch_size = 128

    opt.subcellular_feature_dir = opt.subcellular_feature_dir or os.path.join(
        os.path.dirname(os.path.abspath(opt.data_file)),
        "subcellular_features",
    )

    if opt.resume_run_dir:
        opt.save_path = opt.resume_run_dir
        os.makedirs(opt.save_path, exist_ok=True)
    else:
        # Create a timestamped run directory for this training job (same behavior as stage1).
        run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        opt.save_path = os.path.join(opt.save_path, run_tag)
        os.makedirs(opt.save_path, exist_ok=True)

    args_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args_file = os.path.join(opt.save_path, f'args_resume_{args_stamp}.txt' if opt.resume_run_dir else 'args.txt')
    with open(args_file, 'a') as f:
        json.dump(opt.__dict__, f, indent=2)

    try:
        cell_name_list = np.load(opt.cellname_list, allow_pickle=False)
    except ValueError as e:
        # Compatibility for legacy/object-dtype npy files.
        if "Object arrays cannot be loaded when allow_pickle=False" not in str(e):
            raise
        cell_name_list = np.load(opt.cellname_list, allow_pickle=True)
    cell_name_list = np.asarray(cell_name_list).astype(str).ravel()
    total_cells = len(cell_name_list)
    start_idx = max(1, opt.start_idx)
    end_idx = total_cells if opt.end_idx <= 0 else min(opt.end_idx, total_cells)
    if start_idx > end_idx:
        raise ValueError(f"Invalid index range: start_idx={start_idx}, end_idx={end_idx}, total={total_cells}")
    selected_cells = cell_name_list[start_idx - 1:end_idx]
    print(f"Selected cells index range: [{start_idx}, {end_idx}] / total {total_cells}")

    start_time = time.time()
    skipped_done = 0
    trained_count = 0

    for cellname in selected_cells:
        # current GT net file and current target cell name
        if opt.net_path is not None:
            opt.net_file = os.path.join(opt.net_path, f"{cellname}.csv")
        else:
            opt.net_file = None
        opt.target_cell_name = cellname
        print(cellname, opt.net_file)
        opt.save_name = os.path.join(opt.save_path, cellname)
        rn_out = os.path.join(opt.save_name, f"RN_{opt.n_epochs}.csv")
        if os.path.exists(rn_out):
            print(f"Skip completed cell: {cellname} ({rn_out} exists)")
            skipped_done += 1
            continue

        try:
            os.mkdir(opt.save_name)
            print(f'Create {opt.save_name}')
        except:
            print(f'{opt.save_name} exist')

        model = SC_GRN_model(opt)
        model.train_model()
        trained_count += 1

    end_time = time.time()

    summary_text = (
        f"\nRun Time: {end_time-start_time} s\n"
        f"Selected cells: {len(selected_cells)}\n"
        f"Trained cells: {trained_count}\n"
        f"Skipped completed cells: {skipped_done}\n"
    )
    with open(args_file, 'a') as f:
        f.write(summary_text)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    run()
