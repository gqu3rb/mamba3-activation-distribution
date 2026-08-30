#!/bin/bash
#SBATCH --account=MST115278
#SBATCH --job-name=mamba3_distribution
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --output=/home/m314510193/GithubMamba3Train/mamba3_distribution/mamba3_distribution_%j.log

# Collect the per-block activation distribution of Mamba-3 while it runs a downstream
# task. Same environment contract as mamba3_evals/run_mamba3_lm_eval.sh -- this script only
# swaps the entry point for mamba3_distribution/collect_distribution.py, which wraps that
# same lm_harness_eval.py in a tap.accumulate() block.
#
#   sbatch mamba3_run_ref.sh
#   sbatch --export=ALL,TASKS=hellaswag,LIMIT=512 mamba3_run_ref.sh   # quick smoke run
#   sbatch --export=ALL,MODEL=state-spaces/mamba3-siso-187m,TAG=siso mamba3_run_ref.sh

set -euo pipefail

MODEL="${MODEL:-state-spaces/mamba3-mimo-187m}"
TOKENIZER="${TOKENIZER:-meta-llama/Llama-3.1-8B}"
DTYPE="${DTYPE:-bfloat16}"
ADD_BOS="${ADD_BOS:-true}"
BATCH_SIZE="${BATCH_SIZE:-64}"
CONDA_ENV="${CONDA_ENV:-mamba3}"
# One task by default. The distribution converges in a few hundred documents, so there is
# no reason to pay for the full 7-task suite -- that is what mamba3_evals/ is for.
TASKS="${TASKS:-lambada_openai}"
# lm_eval's own --limit: number of documents per task. Empty means the whole task, which
# is the default so the accuracy/perplexity stay comparable to mamba3_evals/
LIMIT="${LIMIT-}"
# Fold every N-th call at each tap site. 1 costs ~18 extra reductions per block per
# forward; raise it if the job is too slow, the statistics barely move.
TAP_STRIDE="${TAP_STRIDE:-1}"
TAG="${TAG:-$(basename "$MODEL")_${TASKS//,/-}}"

HERE=/home/m314510193/GithubMamba3Train/mamba3_distribution
OUT="${OUT:-$HERE/tap_stats_${TAG}.pt}"
# lm_eval's own results tree (accuracy / perplexity), same convention as
# mamba3_evals/run_mamba3_lm_eval.sh. Without --output_path the metrics exist only as
# printed text in the job log, which nothing downstream can read back.
OUTDIR="${OUTDIR:-$HERE/results_${TAG}}"
# Fixed path, unlike lm_eval's timestamped results_<ISO>.json, so `sbatch --wait` callers
# have something deterministic to read.
METRICS="${METRICS:-$HERE/metrics_${TAG}.json}"

module load miniconda3/26.1.1
module load cuda/13.0
conda activate "$CONDA_ENV"
export PATH="$CONDA_PREFIX/bin:$PATH"

# The env installs mamba_ssm as a *copy*, so edits to the git tree are invisible to
# `import mamba_ssm` unless PYTHONPATH puts the tree ahead of site-packages. The taps live
# in the git tree, so without this the run silently collects nothing.
export MAMBA_REPO=/home/m314510193/GithubMamba3Train/RTX6000_mamba3
export PYTHONPATH="$MAMBA_REPO${PYTHONPATH:+:$PYTHONPATH}"

python - "$MAMBA_REPO" <<'EOF'
import sys, mamba_ssm
expected = sys.argv[1] + "/mamba_ssm/__init__.py"
print("mamba_ssm ->", mamba_ssm.__file__, flush=True)
if mamba_ssm.__file__ != expected:
    sys.exit(f"ERROR: PYTHONPATH shim did not take effect (expected {expected})")
EOF

python -c "import torch; print('torch sees', torch.cuda.device_count(), 'GPU(s)'); \
torch.zeros(1, device='cuda'); print('CUDA alloc OK')"

export HF_HOME=/home/m314510193/GithubMamba3Train/hf_cache
export HF_HUB_OFFLINE=1

LIMIT_ARG=()
if [ -n "$LIMIT" ]; then LIMIT_ARG=(--limit "$LIMIT"); fi

python "$HERE/collect_distribution.py" \
  --tap-out "$OUT" \
  --tap-stride "$TAP_STRIDE" \
  -- \
  --model mamba \
  --model_args "pretrained=${MODEL},tokenizer=${TOKENIZER},dtype=${DTYPE},add_bos_token=${ADD_BOS}" \
  --tasks "$TASKS" \
  --device cuda --batch_size "$BATCH_SIZE" \
  --output_path "$OUTDIR" \
  "${LIMIT_ARG[@]}"

# Print both reports straight into the job log so a run is self-documenting.
python "$HERE/distribution.py" "$OUT" --csv "${OUT%.pt}.csv"
python "$HERE/summarize_results.py" "$OUTDIR" --json "$METRICS"
