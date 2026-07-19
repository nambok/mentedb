//! A tiny, dependency-free management console bundled into the server binary and
//! served at `GET /console`. It polls the aggregate `/metrics` endpoint and renders
//! a live health view (up, uptime, memories, cluster nodes, CPU, memory, request
//! rate and latency), so self-hosters get a management UI with no extra deploy and
//! no build step. It reads only the already-public `/metrics`, so it needs no auth
//! of its own.

use axum::response::Html;

const PAGE: &str = r##"<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>MenteDB console</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin: 0; background: #0a0a0b; color: #e4e4e7; font: 14px/1.5 ui-sans-serif, system-ui, -apple-system, sans-serif; }
  header { padding: 20px 28px; border-bottom: 1px solid #27272a; display: flex; align-items: center; gap: 12px; }
  header h1 { font-size: 16px; font-weight: 600; margin: 0; }
  .dot { width: 10px; height: 10px; border-radius: 50%; background: #52525b; }
  .dot.up { background: #34d399; box-shadow: 0 0 10px rgba(52,211,153,.6); }
  .dot.down { background: #f87171; }
  main { padding: 24px 28px; max-width: 1100px; margin: 0 auto; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 14px; }
  .card { border: 1px solid #27272a; background: #131316; border-radius: 12px; padding: 16px 18px; }
  .card .label { color: #a1a1aa; font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }
  .card .value { font-size: 26px; font-weight: 600; margin-top: 6px; font-variant-numeric: tabular-nums; }
  .card .value.ok { color: #34d399; }
  .card .sub { color: #71717a; font-size: 12px; margin-top: 2px; }
  .muted { color: #71717a; font-size: 12px; margin-top: 20px; }
  a { color: #34d399; text-decoration: none; }
</style>
</head>
<body>
<header>
  <span id="dot" class="dot"></span>
  <h1>MenteDB console</h1>
  <span id="uptime" class="sub" style="color:#71717a"></span>
</header>
<main>
  <div class="grid" id="grid"></div>
  <p class="muted">Live from <a href="/metrics">/metrics</a>, refreshed every 2s. Aggregate only, no per-account data. For full time-series history, scrape with Prometheus and import the Grafana dashboard.</p>
</main>
<script>
function parse(text) {
  const m = {}; // name (no labels) -> summed value
  for (const line of text.split("\n")) {
    if (!line || line[0] === "#") continue;
    const sp = line.lastIndexOf(" ");
    if (sp < 0) continue;
    let key = line.slice(0, sp).trim();
    const val = parseFloat(line.slice(sp + 1));
    if (!isFinite(val)) continue;
    const name = key.indexOf("{") >= 0 ? key.slice(0, key.indexOf("{")) : key;
    m[name] = (m[name] || 0) + val;
  }
  return m;
}
function fmtDur(s) {
  s = Math.floor(s);
  const d = Math.floor(s/86400); s%=86400;
  const h = Math.floor(s/3600); s%=3600;
  const mi = Math.floor(s/60);
  return (d?d+"d ":"") + (h?h+"h ":"") + mi + "m";
}
function fmtBytes(b) {
  if (!b) return "-";
  const u = ["B","KB","MB","GB","TB"]; let i = 0;
  while (b >= 1024 && i < u.length-1) { b/=1024; i++; }
  return b.toFixed(1) + " " + u[i];
}
let prev = null, prevT = 0;
function card(label, value, cls, sub) {
  return `<div class="card"><div class="label">${label}</div><div class="value ${cls||""}">${value}</div>${sub?`<div class="sub">${sub}</div>`:""}</div>`;
}
async function tick() {
  let m, ok = true;
  try { m = parse(await (await fetch("/metrics", {cache:"no-store"})).text()); }
  catch { ok = false; m = {}; }
  const up = ok && m["mentedb_up"] === 1;
  document.getElementById("dot").className = "dot " + (up ? "up" : "down");
  document.getElementById("uptime").textContent = up ? "up " + fmtDur(m["mentedb_uptime_seconds"]||0) : "unreachable";

  const now = Date.now()/1000;
  let reqRate = null, cpu = null;
  if (prev && now > prevT) {
    const dt = now - prevT;
    if (m["mentedb_http_requests_total"] != null && prev["mentedb_http_requests_total"] != null)
      reqRate = Math.max(0, (m["mentedb_http_requests_total"] - prev["mentedb_http_requests_total"]) / dt);
    if (m["process_cpu_seconds_total"] != null && prev["process_cpu_seconds_total"] != null)
      cpu = Math.max(0, (m["process_cpu_seconds_total"] - prev["process_cpu_seconds_total"]) / dt);
  }
  prev = m; prevT = now;

  const cards = [
    card("Status", up ? "Up" : "Down", up ? "ok" : ""),
    card("Memories stored", (m["mentedb_memory_count"]||0).toLocaleString()),
    card("Live cluster nodes", m["mentedb_cluster_live_nodes"] != null ? m["mentedb_cluster_live_nodes"] : "-"),
    card("Requests / sec", reqRate == null ? "…" : reqRate.toFixed(1)),
    card("CPU (cores)", cpu == null ? (isFinite(m["process_cpu_seconds_total"]) ? "…" : "n/a") : cpu.toFixed(2)),
    card("Resident memory", isFinite(m["process_resident_memory_bytes"]) ? fmtBytes(m["process_resident_memory_bytes"]) : "n/a"),
  ];
  document.getElementById("grid").innerHTML = cards.join("");
}
tick(); setInterval(tick, 2000);
</script>
</body>
</html>
"##;

/// `GET /console`: the bundled management console.
pub async fn handler() -> Html<&'static str> {
    Html(PAGE)
}
