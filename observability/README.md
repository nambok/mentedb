# Observability

A one-command Prometheus + Grafana stack for MenteDB. Prometheus scrapes a
server's aggregate `/metrics` endpoint; Grafana starts with the MenteDB dashboard
already provisioned.

## Run it

```bash
# start a server first (metrics are on the same port as the API)
mentedb-server --data-dir ./data

# then bring up the stack
docker compose -f observability/docker-compose.yml up -d
open http://localhost:3000        # Grafana (anonymous admin), "MenteDB" dashboard
```

Grafana comes up at `:3000` (no login), Prometheus at `:9090`. The dashboard shows
up/uptime/memories/live nodes, request rate, latency percentiles, CPU, and resident
memory.

## Point it at your nodes

Edit [`prometheus.yml`](prometheus.yml) and list your targets. The default scrapes
a `mentedb-server` on the host at `:6677`; add more nodes or the platform gateway
(`:8080`) as extra targets. `/metrics` is aggregate only (no per-account labels),
so it is safe to scrape without auth.

## Files

- `prometheus.yml` — scrape config.
- `grafana/provisioning/` — datasource + dashboard providers (auto-loaded).
- `grafana/dashboards/mentedb.json` — the dashboard.
- `docker-compose.yml` — the two services.
