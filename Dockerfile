# eMudhra CA compromise (CVE-2026-111943): the system trust store must not trust
# "emSign ECC Root CA - G3", the only emSign G3 root in the Mozilla set
# (SHA-256 86:A1:EC:BA:08:9C:4A:8D:3B:BE:27:34:C6:12:BA:34:1D:81:3E:04:3C:F9:E8:A8:62:CD:5C:57:A3:6B:BE:6B;
# webpki-roots 1.0.10 removes the same one; emSign G1, C1 and C3 are not affected).
# ca-certificates 20250419~deb12u2 (bookworm-security) already drops it. Every
# Debian stage below requires at least that version, removes the root by exact
# file name, rebuilds the store with update-ca-certificates --fresh, and fails
# the build if these exact certificate bytes (DER, base64) are still in the bundle.
ARG EMSIGN_ECC_G3_DER_B64=MIICTjCCAdOgAwIBAgIKPPYHqWhwDtqLhDAKBggqhkjOPQQDAzBrMQswCQYDVQQGEwJJTjETMBEGA1UECxMKZW1TaWduIFBLSTElMCMGA1UEChMcZU11ZGhyYSBUZWNobm9sb2dpZXMgTGltaXRlZDEgMB4GA1UEAxMXZW1TaWduIEVDQyBSb290IENBIC0gRzMwHhcNMTgwMjE4MTgzMDAwWhcNNDMwMjE4MTgzMDAwWjBrMQswCQYDVQQGEwJJTjETMBEGA1UECxMKZW1TaWduIFBLSTElMCMGA1UEChMcZU11ZGhyYSBUZWNobm9sb2dpZXMgTGltaXRlZDEgMB4GA1UEAxMXZW1TaWduIEVDQyBSb290IENBIC0gRzMwdjAQBgcqhkjOPQIBBgUrgQQAIgNiAAQjpQy4LRL1KPOxst3iAhKAnjlfSU2fySU0WXTsuwYc58Byr+iuL+FBVIcUqEqy6HyC5ltqtdyzdc6LBtCGI79G1Y4PPwT01xySfvalY8L1X44uT6EYGQIrMgqCZH0Wk9GjQjBAMB0GA1UdDgQWBBR8XQKEE9TMipuBzhccLikenEhjQjAOBgNVHQ8BAf8EBAMCAQYwDwYDVR0TAQH/BAUwAwEB/zAKBggqhkjOPQQDAwNpADBmAjEAvvNhzwIQHWSVB7gYboiFBS+DCBeQyh+KTOgNG3qxrdWBCUfvO6wIBHxcmbHtRwfSAjEAnbpV/KlK6O3t5nYBQnvI+GDZjVGLVTv7jHvrZQnD+JbNR6iC8hZVdyR+EhCVBCyj

# Builder stage
# Rust 1.94: several deps (konst, redb) now require >= 1.89; keep headroom so a
# dependency MSRV bump does not break the image build (which CI does not run).
# Pin the builder to the same Debian release as the runtime stage: the
# unsuffixed rust image moved to a newer Debian whose glibc is ahead of the
# runtime's, and a binary linked there fails to start on bookworm.
# Pinned by digest for reproducible builds. Tag: rust:1.94-slim-bookworm
# (multi arch index digest, looked up 2026-10-19). Bump the tag and digest
# together.
FROM rust:1.94-slim-bookworm@sha256:cf9dd0ec73e75f827fe59123fff9dc65af1a1c8363c3c31ee8d7f8ad0b6a5fb2 AS builder

RUN apt-get update && apt-get install -y pkg-config libssl-dev protobuf-compiler ca-certificates && rm -rf /var/lib/apt/lists/*

# Distrust emSign ECC Root CA - G3 (see the top of this file).
ARG EMSIGN_ECC_G3_DER_B64
RUN set -eu; \
    dpkg --compare-versions "$(dpkg-query -W -f='${Version}' ca-certificates)" ge '20250419~deb12u2' \
        || { echo 'ca-certificates is older than 20250419~deb12u2 (bookworm-security)' >&2; exit 1; }; \
    rm -f /usr/share/ca-certificates/mozilla/emSign_ECC_Root_CA_-_G3.crt; \
    sed -i 's|^mozilla/emSign_ECC_Root_CA_-_G3\.crt$|!&|' /etc/ca-certificates.conf; \
    update-ca-certificates --fresh > /dev/null; \
    test "$(grep -c 'BEGIN CERTIFICATE' /etc/ssl/certs/ca-certificates.crt)" -gt 100; \
    if tr -d '\n' < /etc/ssl/certs/ca-certificates.crt | grep -qF "$EMSIGN_ECC_G3_DER_B64"; then \
        echo 'emSign ECC Root CA - G3 is still in the system trust store' >&2; exit 1; \
    fi; \
    if ls /etc/ssl/certs | grep -qF 'emSign_ECC_Root_CA_-_G3'; then \
        echo 'emSign ECC Root CA - G3 is still linked in /etc/ssl/certs' >&2; exit 1; \
    fi

WORKDIR /build
COPY . .

# local-embeddings bundles the Candle model loader so the container has
# zero-config semantic search (model downloads on first start). bedrock adds
# Bedrock extraction and embeddings, authenticated by the task role, so the
# image runs on AWS with no API keys and no model download.
# --locked: build exactly the dependency versions in Cargo.lock.
RUN cargo build --locked --release --bin mentedb-server --features mentedb-server/local-embeddings,mentedb-server/bedrock

# Production stage
# Pinned by digest. Tag: debian:bookworm-slim (multi arch index digest,
# looked up 2026-10-19).
FROM debian:bookworm-slim@sha256:7c7b2c966bc9ee8cedfeef67e0e279108992c77681fa595db4a9d65c06ccc587

# curl backs the HEALTHCHECK below.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libssl3 \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Distrust emSign ECC Root CA - G3 (see the top of this file).
ARG EMSIGN_ECC_G3_DER_B64
RUN set -eu; \
    dpkg --compare-versions "$(dpkg-query -W -f='${Version}' ca-certificates)" ge '20250419~deb12u2' \
        || { echo 'ca-certificates is older than 20250419~deb12u2 (bookworm-security)' >&2; exit 1; }; \
    rm -f /usr/share/ca-certificates/mozilla/emSign_ECC_Root_CA_-_G3.crt; \
    sed -i 's|^mozilla/emSign_ECC_Root_CA_-_G3\.crt$|!&|' /etc/ca-certificates.conf; \
    update-ca-certificates --fresh > /dev/null; \
    test "$(grep -c 'BEGIN CERTIFICATE' /etc/ssl/certs/ca-certificates.crt)" -gt 100; \
    if tr -d '\n' < /etc/ssl/certs/ca-certificates.crt | grep -qF "$EMSIGN_ECC_G3_DER_B64"; then \
        echo 'emSign ECC Root CA - G3 is still in the system trust store' >&2; exit 1; \
    fi; \
    if ls /etc/ssl/certs | grep -qF 'emSign_ECC_Root_CA_-_G3'; then \
        echo 'emSign ECC Root CA - G3 is still linked in /etc/ssl/certs' >&2; exit 1; \
    fi

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
