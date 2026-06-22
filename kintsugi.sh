#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 --cpu|--cuda [--output DIR] <input_file>"
  echo
  echo "  --cpu        Run with CPU image (kintsugi:cpu)"
  echo "  --cuda       Run with CUDA image (kintsugi:cuda)"
  echo "  --output DIR Output directory (default: ./output)"
  exit 1
}

DEVICE=""
OUTPUT_DIR="$(pwd)/output"
INPUT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cpu)    DEVICE=cpu; shift ;;
    --cuda)   DEVICE=cuda; shift ;;
    --output) OUTPUT_DIR="$(realpath "$2")"; shift 2 ;;
    -h|--help) usage ;;
    -*)       echo "Unknown option: $1"; usage ;;
    *)
      [[ -n "$INPUT" ]] && { echo "Unexpected argument: $1"; usage; }
      INPUT="$1"; shift ;;
  esac
done

[[ -z "$DEVICE" || -z "$INPUT" ]] && usage
[[ ! -f "$INPUT" ]] && { echo "Error: file not found: $INPUT"; exit 1; }

INPUT_DIR="$(cd "$(dirname "$INPUT")" && pwd)"
INPUT_FILE="$(basename "$INPUT")"

mkdir -p "$OUTPUT_DIR"

DOCKER_ARGS=(--rm
  --user "$(id -u):$(id -g)"
  --env HOME=/tmp
  -v "${INPUT_DIR}:/app/input:ro"
  -v "${OUTPUT_DIR}:/app/output"
)
[[ "$DEVICE" == "cuda" ]] && DOCKER_ARGS+=(--gpus all)

docker run "${DOCKER_ARGS[@]}" "kintsugi:${DEVICE}" "input/${INPUT_FILE}" "$DEVICE"