# Builder stage
# Rust 1.94: several deps (konst, redb) now require >= 1.89; keep headroom so a
# dependency MSRV bump does not break the image build (which CI does not run).
FROM rust:1.94-slim AS builder

RUN apt-get update && apt-get install -y pkg-config libssl-dev protobuf-compiler && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY . .

# local-embeddings bundles the Candle model loader so the container has
# zero-config semantic search (model downloads on first start).
RUN cargo build --release --bin mentedb-server --features mentedb-server/local-embeddings

# Production stage
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libssl3 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/target/release/mentedb-server /usr/local/bin/mentedb-server

RUN mkdir -p /var/mentedb/data

EXPOSE 8080

CMD ["mentedb-server", "--port", "8080", "--data-dir", "/var/mentedb/data"]
