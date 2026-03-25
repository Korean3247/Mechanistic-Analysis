#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-$REPO_ROOT/.cache/pip}"

mkdir -p \
  "$REPO_ROOT/.cache/huggingface" \
  "$REPO_ROOT/.cache/torch" \
  "$REPO_ROOT/.cache/matplotlib" \
  "$REPO_ROOT/.cache/xdg" \
  "$REPO_ROOT/.cache/pip" \
  "$REPO_ROOT/tmp" \
  "$REPO_ROOT/logs"

cat > "$REPO_ROOT/.env.cloud" <<EOF
export PROJECT_ROOT="$REPO_ROOT"
export HF_HOME="\$PROJECT_ROOT/.cache/huggingface"
export HF_HUB_CACHE="\$HF_HOME/hub"
export HF_DATASETS_CACHE="\$HF_HOME/datasets"
export TRANSFORMERS_CACHE="\$HF_HOME/hub"
export TORCH_HOME="\$PROJECT_ROOT/.cache/torch"
export MPLCONFIGDIR="\$PROJECT_ROOT/.cache/matplotlib"
export XDG_CACHE_HOME="\$PROJECT_ROOT/.cache/xdg"
export TMPDIR="\$PROJECT_ROOT/tmp"
export PIP_CACHE_DIR="\$PROJECT_ROOT/.cache/pip"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTHONNOUSERSITE=1
export WANDB_DISABLED=true
EOF

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip wheel
python -m pip install -e .
python -m pip install datasets

cat <<EOF
Workspace bootstrap complete.

Repo root: $REPO_ROOT
Virtualenv: $VENV_DIR
Env file: $REPO_ROOT/.env.cloud

Next:
  source "$REPO_ROOT/.env.cloud"
  export HF_TOKEN=...
  python scripts/run_cloud_campaign.py --manifest configs/cloud/campaign_neurips_push.yaml --dry-run

Recommended run command:
  python scripts/run_cloud_campaign.py --manifest configs/cloud/campaign_neurips_push.yaml 2>&1 | tee logs/campaign_\$(date +%Y%m%d_%H%M%S).log
EOF
