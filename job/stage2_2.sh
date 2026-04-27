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
  --start_idx 548 \
  --end_idx 1000 \
  --GPU
