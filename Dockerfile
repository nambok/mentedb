# Builder stage
FROM rust:1.85-slim AS builder

RUN apt-get update && apt-get install -y pkg-config libssl-dev && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY . .

RUN cargo build --release --bin mentedb-server

# Production stage
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libssl3 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/target/release/mentedb-server /usr/local/bin/mentedb-server

RUN mkdir -p /var/mentedb/data

EXPOSE 8080

CMD ["mentedb-server", "--host", "0.0.0.0", "--port", "8080", "--data-dir", "/var/mentedb/data"]
