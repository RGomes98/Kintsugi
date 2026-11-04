#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 <command> [options] <input_file>"
  echo
  echo "Commands:"
  echo "  separate   Split a track into drums, bass, vocals and other instruments"
  echo "  master     Remaster a track (EQ, compression, loudness normalization, limiting)"
  echo "  analyze    Measure loudness and peaks without changing the audio"
  echo
  echo "Options:"
  echo "  --cpu              Run with CPU image (kintsugi:cpu) [default]"
  echo "  --cuda             Run with CUDA image (kintsugi:cuda)"
  echo "  --output DIR       Output directory (default: ./output)"
  echo
  echo "Options for master:"
  echo "  --preset NAME      neutral (default), gentle, loud or vintage"
  echo "  --config FILE      TOML file whose values override the preset"
  echo "  --target-lufs X    Loudness target in LUFS, e.g. -14"
  exit 1
}

need_value() {
  [[ $# -ge 2 ]] || { echo "Missing value for $1"; usage; }
}

only_for_master() {
  [[ "$COMMAND" == "master" ]] || { echo "$1 is only valid for the master command"; usage; }
}

[[ $# -eq 0 ]] && usage

COMMAND="$1"
case "$COMMAND" in
  separate|master|analyze) shift ;;
  -h|--help) usage ;;
  *) echo "Unknown command: $COMMAND"; usage ;;
esac

DEVICE=cpu
OUTPUT_DIR="$(pwd)/output"
INPUT=""
CONFIG=""
CLI_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cpu)    DEVICE=cpu; shift ;;
    --cuda)   DEVICE=cuda; shift ;;
    --output) need_value "$@"; OUTPUT_DIR="$(realpath "$2")"; shift 2 ;;
    --preset|--target-lufs)
      only_for_master "$1"; need_value "$@"
      CLI_ARGS+=("$1" "$2"); shift 2 ;;
    --config)
      only_for_master "$1"; need_value "$@"
      [[ -f "$2" ]] || { echo "Error: config file not found: $2"; exit 1; }
      CONFIG="$(realpath "$2")"; shift 2 ;;
    -h|--help) usage ;;
    -*)       echo "Unknown option: $1"; usage ;;
    *)
      [[ -n "$INPUT" ]] && { echo "Unexpected argument: $1"; usage; }
      INPUT="$1"; shift ;;
  esac
done

[[ -z "$INPUT" ]] && usage
[[ ! -f "$INPUT" ]] && { echo "Error: file not found: $INPUT"; exit 1; }

INPUT_DIR="$(cd "$(dirname "$INPUT")" && pwd)"
INPUT_FILE="$(basename "$INPUT")"

DOCKER_ARGS=(--rm
  --user "$(id -u):$(id -g)"
  --env HOME=/tmp
  -v "${INPUT_DIR}:/app/input:ro"
)
[[ "$DEVICE" == "cuda" ]] && DOCKER_ARGS+=(--gpus all)

if [[ "$COMMAND" != "analyze" ]]; then
  mkdir -p "$OUTPUT_DIR"
  DOCKER_ARGS+=(-v "${OUTPUT_DIR}:/app/output")
fi

if [[ -n "$CONFIG" ]]; then
  DOCKER_ARGS+=(-v "$(dirname "$CONFIG"):/app/config:ro")
  CLI_ARGS+=(--config "config/$(basename "$CONFIG")")
fi

case "$COMMAND" in
  separate) CLI_COMMAND=(separate "input/${INPUT_FILE}" --device "$DEVICE") ;;
  master)   CLI_COMMAND=(master "input/${INPUT_FILE}" ${CLI_ARGS[@]+"${CLI_ARGS[@]}"}) ;;
  analyze)  CLI_COMMAND=(analyze "input/${INPUT_FILE}") ;;
esac

docker run "${DOCKER_ARGS[@]}" "kintsugi:${DEVICE}" "${CLI_COMMAND[@]}"
