#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$repo_root/.orx/training.conf"

started_epoch="$(date +%s)"
work_root="$repo_root/.openresearch/work/$METHOD"
official_dir="$work_root/SEED"
model_dir="$work_root/models/qwen3-1.7b"
sft_dir="$work_root/sft_data"
sft_output="$work_root/sft_model"
rl_output="$work_root/rl"
merged_model="$work_root/merged"
log_dir="$work_root/logs"
mkdir -p "$work_root" "$log_dir"

gpu_count="$(python3 - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)"
echo "=== REPRO_CONFIG ==="
echo "method=$METHOD"
echo "model_id=$MODEL_ID"
echo "official_commit=$OFFICIAL_COMMIT"
echo "gpus=$gpu_count"
echo "sft_tasks_per_type=$SFT_TASKS_PER_TYPE"
echo "sft_rollouts_per_task=$SFT_ROLLOUTS_PER_TASK"
echo "train_tasks=$TRAIN_DATA_SIZE"
echo "rollout_group=$GROUP_SIZE"
echo "updates=$TOTAL_UPDATES"
echo "seen_eval_tasks=$FINAL_EVAL_TASKS"
echo "unseen_eval_tasks=$FINAL_EVAL_TASKS"
echo "analyzer_substitution=Qwen/Qwen3-1.7B served locally at temperature 0; deterministic rule fallback"
echo "=== END_REPRO_CONFIG ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
if [[ "$gpu_count" -ne 8 ]]; then
  echo "Expected exactly 8 visible GPUs, found $gpu_count" >&2
  exit 3
fi

if [[ ! -d "$official_dir/.git" ]]; then
  git clone --filter=blob:none "$OFFICIAL_REPO" "$official_dir"
fi
git -C "$official_dir" fetch --depth 1 origin "$OFFICIAL_COMMIT"
git -C "$official_dir" checkout --detach "$OFFICIAL_COMMIT"

ln -sf "$(command -v python3)" /usr/local/bin/python
python3 -m pip install --upgrade pip 'setuptools<80' wheel
python3 -m pip install \
  accelerate codetiming datasets dill hydra-core liger-kernel pandas peft \
  'pyarrow>=19.0.0' pybind11 pylatexenc 'ray[default]>=2.41.0,<=2.50.0' \
  'tensordict>=0.8.0,<=0.10.0,!=0.9.0' \
  torchdata wandb packaging uvicorn fastapi gymnasium==0.29.1 \
  stable-baselines3==2.6.0 alfworld
python3 -m pip install -e "$official_dir" --no-deps
alfworld-download -f

if [[ ! -f "$model_dir/config.json" ]]; then
  MODEL_ID="$MODEL_ID" MODEL_DIR="$model_dir" python3 - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download(repo_id=os.environ["MODEL_ID"], local_dir=os.environ["MODEL_DIR"])
PY
fi

cd "$official_dir"
export ALFWORLD_DATA="${ALFWORLD_DATA:-$HOME/.cache/alfworld}"
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

stage1_log="$log_dir/stage1_vllm.log"
vllm serve "$model_dir" \
  --host 127.0.0.1 --port 60001 \
  --served-model-name qwen3-1.7b \
  --tensor-parallel-size 1 --data-parallel-size 8 \
  --gpu-memory-utilization 0.55 --max-model-len 4096 \
  --seed 260714777 >"$stage1_log" 2>&1 &
vllm_pid=$!
cleanup_vllm() {
  if kill -0 "$vllm_pid" >/dev/null 2>&1; then
    kill "$vllm_pid" >/dev/null 2>&1 || true
    wait "$vllm_pid" >/dev/null 2>&1 || true
  fi
}
trap cleanup_vllm EXIT
for _ in $(seq 1 300); do
  if curl -fsS http://127.0.0.1:60001/v1/models >/dev/null 2>&1; then break; fi
  if ! kill -0 "$vllm_pid" >/dev/null 2>&1; then tail -200 "$stage1_log"; exit 4; fi
  sleep 2
done
curl -fsS http://127.0.0.1:60001/v1/models >/dev/null

python3 scripts/sft/alfworld/pipeline.py \
  --alf-config agent_system/environments/env_package/alfworld/configs/config_tw.yaml \
  --output-dir "$sft_dir" --overwrite \
  --tasks-per-type "$SFT_TASKS_PER_TYPE" \
  --rollouts-per-task "$SFT_ROLLOUTS_PER_TASK" \
  --validation-rollouts 1 --task-batch-size 12 --skill-batch-size 12 \
  --max-steps 15 --history-length 5 --seed 260714777 \
  --request-workers 12 --skill-gen-workers 12 \
  --policy-base-url http://127.0.0.1:60001/v1 --policy-api-key EMPTY \
  --policy-model qwen3-1.7b --policy-temperature 0.0 \
  --policy-max-completion-tokens 384 \
  --skill-base-url http://127.0.0.1:60001/v1 --skill-api-key EMPTY \
  --skill-model qwen3-1.7b --skill-temperature 0.0 \
  --skill-max-completion-tokens 768 \
  --accept-min-delta-count 0 --accept-min-delta-rate -1.0 \
  --sft-val-ratio 0.2

cleanup_vllm
trap - EXIT
python3 "$repo_root/.orx/ensure_sft_data.py" "$sft_dir"
echo "=== STAGE1_METRICS ==="
python3 -m json.tool "$sft_dir/metrics.json" || true
sha256sum "$sft_dir/sft_episode_skill_all.jsonl" "$sft_dir/sft_episode_skill_train.parquet" "$sft_dir/sft_episode_skill_val.parquet"
echo "=== END_STAGE1_METRICS ==="

rm -rf "$sft_output"
torchrun --standalone --nnodes=1 --nproc_per_node=8 \
  -m verl.trainer.fsdp_sft_trainer \
  data.train_files="$sft_dir/sft_episode_skill_train.parquet" \
  data.val_files="$sft_dir/sft_episode_skill_val.parquet" \
  data.train_batch_size=8 data.micro_batch_size_per_gpu=1 \
  data.prompt_key=prompt data.response_key=response \
  'data.prompt_dict_keys=[]' 'data.response_dict_keys=[]' \
  data.max_length=4096 data.truncation=right \
  model.partial_pretrain="$model_dir" \
  model.enable_gradient_checkpointing=True model.strategy=fsdp2 \
  optim.lr=5e-6 trainer.default_local_dir="$sft_output" \
  trainer.project_name=seed-reproduction \
  trainer.experiment_name=qwen3-1.7b-alfworld-public-analyzer-sft \
  "trainer.logger=['console']" trainer.total_epochs=1 trainer.seed=260714777 \
  trainer.default_hdfs_dir=null 2>&1 | tee "$log_dir/sft.log"

sft_checkpoint="$(find "$sft_output" -maxdepth 1 -type d -name 'global_step_*' | sort -V | tail -n 1)"
if [[ -z "$sft_checkpoint" || ! -f "$sft_checkpoint/config.json" ]]; then
  echo "No usable SFT checkpoint found under $sft_output" >&2
  exit 5
fi

export MODEL_PATH="$sft_checkpoint"
export MODELS_ROOT="$work_root/models"
export TRAIN_DATA_SIZE VAL_DATA_SIZE GROUP_SIZE
export PPO_MINI_BATCH_SIZE=64
export PPO_MICRO_BATCH_SIZE_PER_GPU=8
export TENSOR_MODEL_PARALLEL_SIZE=1
export SEED_MODE=mean_std_norm
export SEED_SKILL_MODE=episode_only
export SEED_ANALYSIS_BACKEND=policy_vllm
export SEED_ANALYSIS_NUM_WORKERS=4
export SEED_ANALYSIS_CONTEXT_LENGTH=4096
export SEED_ANALYSIS_MAX_COMPLETION_TOKENS=768
export SEED_ANALYSIS_MAX_MODEL_LEN=4096
export SEED_ANALYSIS_MAX_STEP_SKILLS_PER_TRAJ=0
export SEED_OPD_GATE_BETA=5.0
export SEED_OPD_LOSS_COEF="$OPD_LOSS_COEF"
export SEED_ENABLE_ANALYSIS="$ENABLE_ANALYSIS"
export PROJECT_NAME=seed-reproduction
export EXPERIMENT_NAME="$METHOD-qwen3-1.7b-alfworld"
export DEFAULT_LOCAL_DIR="$rl_output"

rm -rf "$rl_output"
bash examples/seed_trainer/_common/alfworld.sh \
  "trainer.logger=['console']" \
  trainer.total_epochs="$TOTAL_UPDATES" \
  trainer.save_freq=10 trainer.test_freq=5 trainer.val_before_train=True \
  trainer.max_actor_ckpt_to_keep=2 \
  trainer.validation_data_dir="$rl_output/validation" \
  env.alfworld.eval_dataset=eval_in_distribution \
  2>&1 | tee "$log_dir/rl.log"

rl_checkpoint="$(find "$rl_output" -maxdepth 1 -type d -name 'global_step_*' | sort -V | tail -n 1)"
if [[ -z "$rl_checkpoint" || ! -d "$rl_checkpoint/actor" ]]; then
  echo "No RL actor checkpoint found under $rl_output" >&2
  exit 6
fi
rm -rf "$merged_model"
python3 scripts/model_merger.py merge --backend fsdp \
  --local_dir "$rl_checkpoint/actor" --target_dir "$merged_model"

eval_server_log="$log_dir/final_vllm.log"
vllm serve "$merged_model" \
  --host 127.0.0.1 --port 60002 \
  --served-model-name "$METHOD-qwen3-1.7b" \
  --tensor-parallel-size 1 --data-parallel-size 8 \
  --gpu-memory-utilization 0.55 --max-model-len 4096 \
  --seed 260714777 >"$eval_server_log" 2>&1 &
vllm_pid=$!
trap cleanup_vllm EXIT
for _ in $(seq 1 300); do
  if curl -fsS http://127.0.0.1:60002/v1/models >/dev/null 2>&1; then break; fi
  if ! kill -0 "$vllm_pid" >/dev/null 2>&1; then tail -200 "$eval_server_log"; exit 7; fi
  sleep 2
done

for split in eval_in_distribution eval_out_of_distribution; do
  split_dir="$work_root/final_eval/$split"
  mkdir -p "$split_dir"
  python3 -m examples.prompt_agent.local_vllm_alfworld \
    --base-url http://127.0.0.1:60002/v1 --api-key EMPTY \
    --model-name "$METHOD-qwen3-1.7b" \
    --env-num "$FINAL_EVAL_TASKS" --test-times 1 \
    --max-steps 30 --history-length 5 --eval-dataset "$split" \
    --temperature 0.4 --max-completion-tokens 512 \
    --request-workers "$FINAL_EVAL_TASKS" --seed 1 --log-dir "$split_dir"
  result_file="$(find "$split_dir/results" -type f -name 'results_*.json' | sort | tail -n 1)"
  echo "=== FINAL_${split}_JSON ==="
  python3 -m json.tool "$result_file"
  echo "=== END_FINAL_${split}_JSON ==="
done

cleanup_vllm
trap - EXIT
echo "=== EARLY_TRAINING_METRIC_LINES ==="
grep -E "val/.*/test_score|val/subtask_success_rate_mean|training/global_step|actor/opd_loss" "$log_dir/rl.log" || true
echo "=== END_EARLY_TRAINING_METRIC_LINES ==="
ended_epoch="$(date +%s)"
python3 - "$started_epoch" "$ended_epoch" <<'PY'
import sys
elapsed = int(sys.argv[2]) - int(sys.argv[1])
print(f"ACTUAL_ELAPSED_SECONDS={elapsed}")
print(f"ACTUAL_ELAPSED_HOURS={elapsed / 3600:.6f}")
PY
