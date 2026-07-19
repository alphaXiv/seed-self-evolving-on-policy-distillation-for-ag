#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
config_file="$repo_root/.orx/reproduction.conf"

if [[ ! -f "$config_file" ]]; then
  echo "Missing tracked experiment config: $config_file" >&2
  exit 2
fi

# shellcheck disable=SC1090
source "$config_file"

required=(
  OFFICIAL_REPO OFFICIAL_COMMIT MODEL_ID MODEL_LABEL EVAL_DATASET ENV_NUM
  TEST_TIMES MAX_STEPS HISTORY_LENGTH TEMPERATURE MAX_COMPLETION_TOKENS
  ENV_SEED TENSOR_PARALLEL_SIZE DATA_PARALLEL_SIZE GPU_MEMORY_UTILIZATION
  MAX_MODEL_LEN
)
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "Missing required config value: $name" >&2
    exit 2
  fi
done

visible_gpu_count="$(python3 - <<'PY'
try:
    import torch
    print(torch.cuda.device_count())
except Exception:
    print(0)
PY
)"

echo "=== SEED_REPRO_CONFIG ==="
echo "official_repo=$OFFICIAL_REPO"
echo "official_commit=$OFFICIAL_COMMIT"
echo "model_id=$MODEL_ID"
echo "model_label=$MODEL_LABEL"
echo "eval_dataset=$EVAL_DATASET"
echo "env_num=$ENV_NUM"
echo "test_times=$TEST_TIMES"
echo "max_steps=$MAX_STEPS"
echo "history_length=$HISTORY_LENGTH"
echo "temperature=$TEMPERATURE"
echo "max_completion_tokens=$MAX_COMPLETION_TOKENS"
echo "env_seed=$ENV_SEED"
echo "tensor_parallel_size=$TENSOR_PARALLEL_SIZE"
echo "data_parallel_size=$DATA_PARALLEL_SIZE"
echo "visible_gpu_count=$visible_gpu_count"
echo "=== END_SEED_REPRO_CONFIG ==="

if (( visible_gpu_count < DATA_PARALLEL_SIZE )); then
  echo "Configured for $DATA_PARALLEL_SIZE GPUs but only $visible_gpu_count are visible." >&2
  exit 3
fi

work_dir="$repo_root/.openresearch/work"
artifact_dir="$repo_root/.openresearch/artifacts"
source_dir="$work_dir/seed-official"
model_dir="$work_dir/models/$MODEL_LABEL"
eval_dir="$artifact_dir/$MODEL_LABEL"
mkdir -p "$work_dir" "$artifact_dir" "$(dirname "$model_dir")" "$eval_dir"

if [[ ! -d "$source_dir/.git" ]]; then
  git clone --filter=blob:none "$OFFICIAL_REPO" "$source_dir"
fi
git -C "$source_dir" fetch --depth 1 origin "$OFFICIAL_COMMIT"
git -C "$source_dir" checkout --detach "$OFFICIAL_COMMIT"

ln -sf "$(command -v python3)" /usr/local/bin/python
python3 -m pip install --upgrade pip 'setuptools<80' wheel
python3 -m pip install \
  "vllm==0.11.0" \
  "gymnasium==0.29.1" \
  "stable-baselines3==2.6.0" \
  alfworld
python3 -m pip install -e "$source_dir"

if ! command -v alfworld-download >/dev/null 2>&1; then
  echo "alfworld-download was not installed" >&2
  exit 4
fi
alfworld-download -f

if [[ ! -f "$model_dir/config.json" ]]; then
  MODEL_ID="$MODEL_ID" MODEL_DIR="$model_dir" python3 - <<'PY'
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id=os.environ["MODEL_ID"],
    local_dir=os.environ["MODEL_DIR"],
)
PY
fi

cd "$source_dir"
export MODEL_PATH="$model_dir"
export MODEL_NAME="$MODEL_LABEL"
export TENSOR_PARALLEL_SIZE
export DATA_PARALLEL_SIZE
export GPU_MEMORY_UTILIZATION
export MAX_MODEL_LEN
export ENV_NUM
export TEST_TIMES
export MAX_STEPS
export HISTORY_LENGTH
export EVAL_DATASET
export TEMPERATURE
export MAX_COMPLETION_TOKENS
export REQUEST_WORKERS="$ENV_NUM"
export EVAL_LOG_DIR="$eval_dir"

bash examples/prompt_agent/run_local_vllm_alfworld.sh --seed "$ENV_SEED"

result_file="$(find "$eval_dir/results" -type f -name 'results_*.json' -print | sort | tail -n 1)"
if [[ -z "$result_file" ]]; then
  echo "Official evaluator completed without a result JSON." >&2
  exit 5
fi

echo "=== SEED_REPRO_EVIDENCE_JSON ==="
python3 -m json.tool "$result_file"
echo "=== END_SEED_REPRO_EVIDENCE_JSON ==="
