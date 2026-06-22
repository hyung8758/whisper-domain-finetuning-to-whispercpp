#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG_PATH="${1:-}"
EXP_NAME="${EXP_NAME:-}"
NPROC_PER_NODE="${NPROC_PER_NODE:-}"
MASTER_PORT="${MASTER_PORT:-29500}"

if [[ "$CONFIG_PATH" != *.yaml && "$CONFIG_PATH" != *.yml ]]; then
  echo "Config file must have .yaml or .yml extension: $CONFIG_PATH" >&2
  exit 1
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config yaml file not found: $CONFIG_PATH" >&2
  exit 1
fi

if [[ -z "$NPROC_PER_NODE" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    NPROC_PER_NODE="$(nvidia-smi --list-gpus | wc -l | tr -d ' ')"
  else
    NPROC_PER_NODE="1"
  fi
fi


command=(
  torchrun
  --standalone
  --nproc_per_node="$NPROC_PER_NODE"
  --master_port="$MASTER_PORT"
  scripts/train_whisper_lora.py
  --config "$CONFIG_PATH"
)

if [[ -n "$EXP_NAME" ]]; then
  command+=(--exp_name "$EXP_NAME")
fi

echo "CONFIG_PATH=$CONFIG_PATH"
echo "EXP_NAME=${EXP_NAME:-<config default>}"
echo "NPROC_PER_NODE=$NPROC_PER_NODE"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<all visible GPUs>}"
echo "MASTER_PORT=$MASTER_PORT"
echo "${command[*]}"

"${command[@]}"
