# Builder stage
# Rust 1.94: several deps (konst, redb) now require >= 1.89; keep headroom so a
# dependency MSRV bump does not break the image build (which CI does not run).
# Pin the builder to the same Debian release as the runtime stage: the
# unsuffixed rust image moved to a newer Debian whose glibc is ahead of the
# runtime's, and a binary linked there fails to start on bookworm.
FROM rust:1.94-slim-bookworm AS builder

RUN apt-get update && apt-get install -y pkg-config libssl-dev protobuf-compiler && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY . .

# local-embeddings bundles the Candle model loader so the container has
# zero-config semantic search (model downloads on first start). bedrock adds
# Bedrock extraction and embeddings, authenticated by the task role, so the
# image runs on AWS with no API keys and no model download.
RUN cargo build --release --bin mentedb-server --features mentedb-server/local-embeddings,mentedb-server/bedrock

# Production stage
FROM debian:bookworm-slim

# curl backs the HEALTHCHECK below.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libssl3 \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Run as a fixed non-root uid/gid so volume ownership is predictable across
# hosts and orchestrators (chown a bind mount to 10001:10001).
ARG MENTEDB_UID=10001
ARG MENTEDB_GID=10001
RUN groupadd --system --gid ${MENTEDB_GID} mentedb \
    && useradd --system --uid ${MENTEDB_UID} --gid ${MENTEDB_GID} \
       --home-dir /var/mentedb --no-create-home --shell /usr/sbin/nologin mentedb \
    && mkdir -p /var/mentedb/data /var/mentedb/.cache \
    && chown -R ${MENTEDB_UID}:${MENTEDB_GID} /var/mentedb

COPY --from=builder /build/target/release/mentedb-server /usr/local/bin/mentedb-server

# The Candle model (local-embeddings) downloads on first start into
# $HOME/.cache/huggingface (the hf-hub default cache, which does not read
# HF_HOME), so HOME must be a directory the service user can write. Mount a
# volume at /var/mentedb/.cache to keep the model across container restarts.
ENV HOME=/var/mentedb

USER ${MENTEDB_UID}:${MENTEDB_GID}
WORKDIR /var/mentedb

# 6677 = REST API, /metrics, and /console; 6678 = gRPC (both start by default).
# This is the default port the CLI, docs, and the observability compose all assume.
EXPOSE 6677 6678

# /v1/ready answers once the server can take traffic. When TLS is on
# (MENTEDB_TLS_CERT_FILE / MENTEDB_TLS_KEY_FILE), override this check with
# https and --insecure, or point it at a name the certificate covers.
# Works with and without TLS: plain HTTP first, then HTTPS. -k only skips
# certificate verification for this loopback probe; clients still verify.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://localhost:6677/v1/ready || curl -fsSk https://localhost:6677/v1/ready || exit 1

CMD ["mentedb-server", "--port", "6677", "--data-dir", "/var/mentedb/data"]
