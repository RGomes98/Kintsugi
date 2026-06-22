#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 [--cpu] [--cuda]"
  echo
  echo "  --cpu   Build the CPU image (kintsugi:cpu)"
  echo "  --cuda  Build the CUDA image (kintsugi:cuda)"
  echo
  echo "  With no flags, both images are built."
  exit 1
}

BUILD_CPU=false
BUILD_CUDA=false

if [[ $# -eq 0 ]]; then
  BUILD_CPU=true
  BUILD_CUDA=true
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cpu)     BUILD_CPU=true; shift ;;
    --cuda)    BUILD_CUDA=true; shift ;;
    -h|--help) usage ;;
    *)         echo "Unknown option: $1"; usage ;;
  esac
done

build_image() {
  local target="$1"   # cpu or cuda

  echo "==> [1/3] Building exporter image (kintsugi:exporter-${target}) ..."
  docker build --build-arg TARGET="${target}" --target exporter \
    -t "kintsugi:exporter-${target}" .

  echo "==> [2/3] Exporting ${target^^} model via docker run ..."
  mkdir -p python/models
  local run_args=(--rm -v "$(pwd)/python/models:/app/python/models")
  [[ "$target" == "cuda" ]] && run_args+=(--gpus all)
  docker run "${run_args[@]}" "kintsugi:exporter-${target}" --device "${target}"

  echo "==> [3/3] Building kintsugi:${target} ..."
  docker build --build-arg TARGET="${target}" -t "kintsugi:${target}" .

  echo "==> Cleaning up exporter image ..."
  docker rmi "kintsugi:exporter-${target}" 2>/dev/null || true

  echo "==> kintsugi:${target} is ready."
}

$BUILD_CPU  && build_image cpu
$BUILD_CUDA && build_image cuda