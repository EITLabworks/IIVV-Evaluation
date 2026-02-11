#!/usr/bin/env bash

# Start script:
#chmod +x start_gpu_tuning_tmux.sh
#./start_gpu_tuning_tmux.sh


# ---- CONFIG ----
ENV_NAME="rapids"
SCRIPT="2_model_gpu_tuning.py"

SESSION1="GPUmodeltuning_hvc_f1"
SESSION2="GPUmodeltuning_cvhrvlr_f1"
SESSION3="GPUmodeltuning_mc_f1"

# ---- helper: conda init for non-interactive shells ----
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"

# ---- Session 1 ----
tmux new-session -d -s $SESSION1
tmux send-keys -t $SESSION1 "conda activate $ENV_NAME" C-m
tmux send-keys -t $SESSION1 "CUDA_VISIBLE_DEVICES=5 python $SCRIPT --model all --task healthy_vs_cancer --metric f1" C-m

# ---- Session 2 ----
tmux new-session -d -s $SESSION2
tmux send-keys -t $SESSION2 "conda activate $ENV_NAME" C-m
tmux send-keys -t $SESSION2 "CUDA_VISIBLE_DEVICES=6 python $SCRIPT --model all --task cancer_vs_lr_hr --metric f1" C-m

# ---- Session 3 ----
tmux new-session -d -s $SESSION3
tmux send-keys -t $SESSION3 "conda activate $ENV_NAME" C-m
tmux send-keys -t $SESSION3 "CUDA_VISIBLE_DEVICES=7 python $SCRIPT --model all --task MC --metric f1" C-m

echo "✅ All tmux sessions started."
echo "Attach with:"
echo "tmux attach -t $SESSION1"
echo "tmux attach -t $SESSION2"
echo "tmux attach -t $SESSION3"
