ARG TARGET=cpu

FROM ubuntu:22.04 AS exporter-base-cpu
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS exporter-base-cuda
FROM exporter-base-${TARGET} AS exporter
ARG TARGET=cpu

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3.11-dev \
    && rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/venv
RUN python3.11 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY python/requirements-${TARGET}.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /app
COPY python/ ./python/
ENTRYPOINT ["python", "-u", "python/scripts/export_model.py"]

FROM ubuntu:22.04 AS builder
ARG TARGET=cpu

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential pkg-config libssl-dev \
    python3.11 python3.11-venv python3.11-dev \
    && rm -rf /var/lib/apt/lists/*

COPY --from=exporter /opt/venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# tch-rs locates LibTorch by importing torch; expose the venv's libs to the linker.
ENV LD_LIBRARY_PATH="/opt/venv/lib/python3.11/site-packages/torch/lib"

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path
ENV PATH="/root/.cargo/bin:$PATH"

WORKDIR /app
COPY . .
ENV LIBTORCH_USE_PYTORCH=1
RUN cargo build --release

# Skip libcublas* and libcudnn* — already provided by the nvidia runtime base image.
RUN mkdir -p /opt/torch/lib && \
    find /opt/venv/lib/python3.11/site-packages/torch/lib/ \
      -maxdepth 1 -name '*.so*' ! -name 'libcublas*' ! -name 'libcudnn*' \
      -exec cp -P {} /opt/torch/lib/ \;

FROM ubuntu:22.04 AS runtime-base-cpu
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS runtime-base-cuda
FROM runtime-base-${TARGET} AS runtime
ARG TARGET=cpu

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libssl3 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /opt/torch/lib ./torch/lib
COPY --from=builder /app/python/models ./python/models
COPY --from=builder /app/target/release/cli ./cli

ENV LD_LIBRARY_PATH="/app/torch/lib:/usr/local/cuda-12.1/lib64:/usr/lib/x86_64-linux-gnu"
ENTRYPOINT ["./cli"]