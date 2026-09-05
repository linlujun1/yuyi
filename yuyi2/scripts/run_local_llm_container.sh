#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-quay.io/ascend/vllm-ascend:v0.22.1rc1}"
CONTAINER_NAME="${CONTAINER_NAME:-yuyi2-qwen3-1_7b-openai}"
MODEL_PATH="${MODEL_PATH:-/user_home/linlujun/linlujun/model/Qwen3-1.7B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3-1.7B}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
NPU_DEVICES="${NPU_DEVICES:-5}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1536}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.35}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-768}"
CONTAINER_MODEL_PATH="/models/${SERVED_MODEL_NAME}"

cleanup() {
  echo
  echo "[yuyi2] stopping ${CONTAINER_NAME} ..."
  docker stop --time 30 "${CONTAINER_NAME}" >/dev/null 2>&1 || true
  docker rm "${CONTAINER_NAME}" >/dev/null 2>&1 || true
  echo "[yuyi2] container removed; NPU memory should be released."
}

trap cleanup INT TERM EXIT

if [ ! -d "${MODEL_PATH}" ]; then
  echo "[yuyi2] model path does not exist: ${MODEL_PATH}" >&2
  exit 1
fi

if docker ps -a --format '{{.Names}}' | grep -Fxq "${CONTAINER_NAME}"; then
  echo "[yuyi2] removing existing container: ${CONTAINER_NAME}"
  docker stop --time 10 "${CONTAINER_NAME}" >/dev/null 2>&1 || true
  docker rm "${CONTAINER_NAME}" >/dev/null 2>&1 || true
fi

echo "[yuyi2] image: ${IMAGE}"
echo "[yuyi2] model: ${MODEL_PATH}"
echo "[yuyi2] served model: ${SERVED_MODEL_NAME}"
echo "[yuyi2] NPU devices: ${NPU_DEVICES}"
echo "[yuyi2] OpenAI URL: http://${HOST}:${PORT}/v1"
echo "[yuyi2] max model len: ${MAX_MODEL_LEN}"
echo "[yuyi2] gpu memory utilization: ${GPU_MEMORY_UTILIZATION}"

docker run -d \
  --name "${CONTAINER_NAME}" \
  --privileged \
  --ipc=host \
  --shm-size=16g \
  -p "${HOST}:${PORT}:8000" \
  -e ASCEND_RT_VISIBLE_DEVICES="${NPU_DEVICES}" \
  -e VLLM_USE_V1=0 \
  -e VLLM_USE_MODELSCOPE=false \
  -e PYTORCH_NPU_ALLOC_CONF=expandable_segments:True \
  -v "${MODEL_PATH}:${CONTAINER_MODEL_PATH}:ro" \
  "${IMAGE}" \
  python -m vllm.entrypoints.openai.api_server \
    --model "${CONTAINER_MODEL_PATH}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code \
    --dtype bfloat16 \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
    --enforce-eager

echo "[yuyi2] waiting for server logs. Press Ctrl+C to stop and remove the container."
docker logs -f "${CONTAINER_NAME}"
