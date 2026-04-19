# ── Stage 1: Build llama-server with CUDA ──────────────────────────────
FROM nvidia/cuda:12.8.1-devel-ubuntu22.04 AS llama-build

ARG LLAMA_CPP_REF=master
# Common CUDA architectures covering the GPUs we actually see on Vast.ai:
#   80  = A100, A30 (Ampere datacenter)
#   86  = RTX 30xx, A10, A40, A6000 (Ampere consumer)
#   89  = RTX 4090, L40, L40S, RTX 6000 Ada (Ada Lovelace)
#   90  = H100, H200 (Hopper)
#   100 = B200 (Blackwell)
#   120 = B200 Ultra (Blackwell Ultra)
# Previous list omitted 80 — llama-server then aborts with "no kernel
# image is available for execution on the device" on A100 hosts.
ARG CUDA_ARCHITECTURES="80;86;89;90;100;120"

RUN apt-get update -qq && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq git build-essential curl && \
    curl -fsSL https://github.com/Kitware/CMake/releases/download/v3.31.6/cmake-3.31.6-linux-x86_64.tar.gz \
      | tar -xzf - -C /usr/local --strip-components=1

RUN git clone --depth 1 --branch "$LLAMA_CPP_REF" https://github.com/ggml-org/llama.cpp /opt/llama.cpp

WORKDIR /opt/llama.cpp
RUN cmake -B build \
      -DGGML_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHITECTURES" \
      -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build --config Release -j"$(nproc)" -t llama-server

# ── Stage 2: Build tanrenai-gpu ────────────────────────────────────────
FROM golang:1.25-bookworm AS go-build

WORKDIR /src
COPY . .
RUN CGO_ENABLED=0 go build -o /tanrenai-gpu .

# ── Stage 3: Runtime image ─────────────────────────────────────────────
FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04

RUN apt-get update -qq && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ca-certificates curl libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# llama-server binary and shared libs
COPY --from=llama-build /opt/llama.cpp/build/bin/llama-server /usr/local/bin/llama-server
COPY --from=llama-build /opt/llama.cpp/build/bin/*.so* /usr/local/lib/
RUN ldconfig

# tanrenai-gpu binary
COPY --from=go-build /tanrenai-gpu /usr/local/bin/tanrenai-gpu

# Set up tanrenai data dirs
RUN mkdir -p /root/.local/share/tanrenai/bin && \
    ln -s /usr/local/bin/llama-server /root/.local/share/tanrenai/bin/llama-server

ENV LD_LIBRARY_PATH=/usr/local/lib
EXPOSE 11435

ENTRYPOINT ["tanrenai-gpu"]
CMD ["serve", "--host", "0.0.0.0", "--port", "11435"]
