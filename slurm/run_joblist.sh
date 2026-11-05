#!/bin/bash
# Generic SLURM array template: array task i runs line (i+1) of the job list file $LIST,
# executed as a shell command from the repository root.
#
# Submit from the repository root:
#   N=$(wc -l < jobs/X.txt)
#   sbatch --array=0-$((N-1)) --job-name=X --export=ALL,LIST=jobs/X.txt slurm/run_joblist.sh
# LIST must be a path without commas, because --export splits its argument at commas.
# The log directory (logs/) must exist before submission.
#
# Cluster-specific options are marked with <PLACEHOLDER>; replace them before use.
# <PARTITION>: name of a partition (queue) with GPU nodes on your cluster.
#SBATCH --partition=<PARTITION>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
# Optional: to exclude unreliable nodes, remove one '#' below and list the nodes (comma-separated).
##SBATCH --exclude=<NODE_LIST>
#SBATCH --time=04:00:00
#SBATCH --output=logs/jl_%x_%A_%a.out
#SBATCH --error=logs/jl_%x_%A_%a.err
set -e
# Run from the directory the job was submitted from (the repository root).
cd "${SLURM_SUBMIT_DIR}"
# <ENVIRONMENT>: activate a Python environment with PyTorch and CUDA support, for example
#   source <CONDA_INSTALL_DIR>/etc/profile.d/conda.sh
#   conda activate <ENV_NAME>
# Stop if PyTorch does not see a GPU.
python -c 'import torch,sys; ok=torch.cuda.is_available(); print("cuda", ok); sys.exit(0 if ok else 3)' || { echo "CUDA not available on this node, exiting"; exit 3; }
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "${LIST}")
echo "== ${LIST} #${SLURM_ARRAY_TASK_ID}: ${LINE}  $(date) on $(hostname) =="
bash -c "${LINE}"
echo "== DONE $(date) =="
