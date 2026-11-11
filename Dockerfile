# syntax=docker/dockerfile:1
ARG CUDA_VERSION=12.4.1
ARG UBUNTU=ubuntu22.04
ARG WHISPER_CPP_VERSION=v1.7.6

# ---------------------------------------------------------------------------
# whisper.cpp server: plain CPU build (fallback when no GPU is visible)
# ---------------------------------------------------------------------------
FROM ubuntu:22.04 AS whisper-cpu
ARG WHISPER_CPP_VERSION
RUN apt-get update && apt-get install -y --no-install-recommends git cmake build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 --branch ${WHISPER_CPP_VERSION} https://github.com/ggml-org/whisper.cpp /src
WORKDIR /src
RUN cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DGGML_NATIVE=OFF -DWHISPER_BUILD_TESTS=OFF \
    && cmake --build build -j"$(nproc)" --target whisper-server \
    && cp build/bin/whisper-server /whisper-server-cpu

# ---------------------------------------------------------------------------
# whisper.cpp server: CUDA build
# ---------------------------------------------------------------------------
FROM nvidia/cuda:${CUDA_VERSION}-devel-${UBUNTU} AS whisper-cuda
ARG WHISPER_CPP_VERSION
# 61=Pascal(GTX10xx) 75=Turing(RTX20xx/GTX16xx) 86=Ampere(RTX30xx) 89=Ada(RTX40xx)
ARG CUDA_ARCHS="61;75;86;89"
RUN apt-get update && apt-get install -y --no-install-recommends git cmake build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 --branch ${WHISPER_CPP_VERSION} https://github.com/ggml-org/whisper.cpp /src
WORKDIR /src
RUN cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DGGML_NATIVE=OFF -DWHISPER_BUILD_TESTS=OFF \
        -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}" \
    && cmake --build build -j"$(nproc)" --target whisper-server \
    && cp build/bin/whisper-server /whisper-server-cuda

# ---------------------------------------------------------------------------
# Go application (pure Go, static binary)
# ---------------------------------------------------------------------------
FROM golang:1.22-bookworm AS app
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY cmd ./cmd
COPY internal ./internal
ARG VERSION=dev
RUN CGO_ENABLED=0 go build -trimpath -ldflags "-s -w -X main.version=${VERSION}" -o /captionsync ./cmd/captionsync

# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------
FROM nvidia/cuda:${CUDA_VERSION}-runtime-${UBUNTU}
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg ca-certificates tzdata \
    && rm -rf /var/lib/apt/lists/*
COPY --from=whisper-cpu /whisper-server-cpu /usr/local/bin/
COPY --from=whisper-cuda /whisper-server-cuda /usr/local/bin/
COPY --from=app /captionsync /usr/local/bin/captionsync
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    CONFIG_DIR=/config \
    MEDIA_ROOTS=/media \
    PORT=8080 \
    PUID=99 \
    PGID=100 \
    UMASK=002

VOLUME ["/config"]
EXPOSE 8080
ENTRYPOINT ["/entrypoint.sh"]
