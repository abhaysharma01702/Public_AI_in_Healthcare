#!/bin/bash
# One-time Python environment for the quality module on IIIT Ada. Runs as a short CPU-only job, so
# nothing heavy runs on the gateway. Installs Miniconda's Python (no conda channels used) + a venv with the same
# PyTorch as the Kaggle run (torch 2.11.0 / torchvision 0.26.0) into $HOME; --no-cache-dir keeps the 25 GB home quota free.
#SBATCH --job-name=qm_setup
#SBATCH --account=<your_account>
#SBATCH --qos=<your_qos>
#SBATCH --partition=u22
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:45:00
#SBATCH --exclude=gnode004
#SBATCH --output=$HOME/project/logs/slurm_%x_%j.out

set -euo pipefail
echo "node $(hostname)  start $(date -Is)"
curl -sI --max-time 15 https://pypi.org | head -1 || { echo "no internet on this node"; exit 2; }
if [ ! -x "$HOME/miniconda3/bin/python3" ]; then
  tmp=$(mktemp -d)
  curl -sSL -o "$tmp/mc.sh" https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash "$tmp/mc.sh" -b -p "$HOME/miniconda3"
  rm -rf "$tmp"
fi
rm -rf "$HOME/envs/qm"                                   # rebuild cleanly (first attempt pulled a CUDA 13.0 build)
"$HOME/miniconda3/bin/python3" -m venv "$HOME/envs/qm"
source "$HOME/envs/qm/bin/activate"
pip install --no-cache-dir --upgrade pip
# Ada driver 570 supports CUDA <= 12.8: take the cu128 build from the official PyTorch index (PyPI's default is cu130)
pip install --no-cache-dir torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install --no-cache-dir "pandas==2.3.3" "numpy<2.5" scipy pyyaml pillow    # pandas as on Kaggle (2.3.3)
python -c "import torch, torchvision, pandas, numpy, scipy, yaml, PIL; print('torch', torch.__version__, 'cuda build', torch.version.cuda, 'tv', torchvision.__version__, 'pandas', pandas.__version__, 'numpy', numpy.__version__)"
du -sh "$HOME/miniconda3" "$HOME/envs/qm"
echo "end $(date -Is)"
