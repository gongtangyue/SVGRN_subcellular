#!/bin/bash
#SBATCH --account=bcod-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --job-name=svgrn_1
#SBATCH --output=job/svgrn_1_%j.out

cd /u/tgong/SVGRN_subcellular

# Delta RH9: submit this job from an already-activated environment.
echo "python: $(which python)"
python --version

python ./main_stage2_allcell.py \
  --setting default \
  --data_file ./in_sim/g110_c2k_n01/expression_loc_cluster_wlayout.csv \
  --cellname_list ./in_sim/g110_c2k_n01/cellname_list_all.npy \
  --model_file ./runs/stage1/g110_c2k_n01/20260407_161130/stage1.pt \
  --net_path ./in_sim/g110_c2k_n01/cell_specific_GRN \
  --subcellular_feature_dir ./in_sim/g110_c2k_n01/r01_features \
  --resume_run_dir ./runs/stage2/g110_c2k_n01/20260407 \
  --start_idx 49 \
  --end_idx 500 \
  --GPU

#!/bin/bash
#
#SBATCH --job-name=4k_3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4  # could be 1 for py-torch
#SBATCH --gpus-per-node=1
#SBATCH --mem=40G
#SBATCH --partition=gpuA40x4,gpuA100x4
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8   # spread out to use 1 core per numa, set to 64 if tasks is 1
#SBATCH --constraint="scratch"
#SBATCH --gpu-bind=closest   # select a cpu close to gpu on pci bus topology
#SBATCH --account=bcod-delta-gpu    # <- match to a "Project" returned by the "accounts" command
#SBATCH --exclusive  # dedicated node for this job
#SBATCH --no-requeue
#SBATCH --output=/projects/bcod/yurui/spatial_GRN/log/%x_%j.out
#SBATCH --error=/projects/bcod/yurui/spatial_GRN/log/%x_%j.err