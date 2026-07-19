//! A tiny, dependency-free management console bundled into the server binary and
//! served at `GET /console`. It has two tabs: Health (polls the public `/metrics`)
//! and Memories (browses and deletes memories via the admin-key-gated
//! `/v1/admin/memories` endpoints). Self-hosters get a real DB admin UI with no
//! extra deploy and no build step.

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
  header { padding: 18px 28px; border-bottom: 1px solid #27272a; display: flex; align-items: center; gap: 12px; }
  header h1 { font-size: 16px; font-weight: 600; margin: 0; }
  .dot { width: 10px; height: 10px; border-radius: 50%; background: #52525b; }
  .dot.up { background: #34d399; box-shadow: 0 0 10px rgba(52,211,153,.6); }
  .dot.down { background: #f87171; }
  nav { display: flex; gap: 4px; padding: 0 22px; border-bottom: 1px solid #27272a; }
  nav button { background: none; border: none; color: #a1a1aa; padding: 12px 14px; font: inherit; cursor: pointer; border-bottom: 2px solid transparent; }
  nav button.active { color: #e4e4e7; border-bottom-color: #34d399; }
  main { padding: 24px 28px; max-width: 1200px; margin: 0 auto; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 14px; }
  .card { border: 1px solid #27272a; background: #131316; border-radius: 12px; padding: 16px 18px; }
  .card .label { color: #a1a1aa; font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }
  .card .value { font-size: 26px; font-weight: 600; margin-top: 6px; font-variant-numeric: tabular-nums; }
  .card .value.ok { color: #34d399; }
  .muted { color: #71717a; font-size: 12px; margin-top: 20px; }
  a { color: #34d399; text-decoration: none; }
  .sub { color: #71717a; font-size: 12px; }
  .row { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; margin-bottom: 14px; }
  input, select, button.act { background: #131316; border: 1px solid #27272a; color: #e4e4e7; border-radius: 8px; padding: 8px 10px; font: inherit; }
  button.act { cursor: pointer; }
  button.act:hover { border-color: #3f3f46; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid #1f1f23; vertical-align: top; }
  th { color: #a1a1aa; font-weight: 500; font-size: 12px; text-transform: uppercase; letter-spacing: .03em; }
  td.content { max-width: 520px; }
  .pill { font-size: 11px; padding: 1px 7px; border-radius: 999px; background: #1f1f23; color: #a1a1aa; }
  .mono { font-family: ui-monospace, SFMono-Regular, monospace; color: #71717a; font-size: 12px; }
  .del { color: #f87171; cursor: pointer; background: none; border: none; font: inherit; }
  .hidden { display: none; }
</style>
</head>
<body>
<header>
  <span id="dot" class="dot"></span>
  <h1>MenteDB console</h1>
  <span id="uptime" class="sub"></span>
</header>
<nav>
  <button id="tab-health" class="active" onclick="showTab('health')">Health</button>
  <button id="tab-memories" onclick="showTab('memories')">Memories</button>
  <button id="tab-query" onclick="showTab('query')">Query</button>
</nav>
<main>
  <section id="health">
    <div class="grid" id="grid"></div>
    <p class="muted">Live from <a href="/metrics">/metrics</a>, refreshed every 2s. Aggregate only, no per-account data. For history, scrape with Prometheus and import the Grafana dashboard.</p>
  </section>
  <section id="memories" class="hidden">
    <div class="row">
      <input id="key" type="password" placeholder="admin key (x-api-key)" style="width:220px" />
      <input id="q" placeholder="search content" style="width:200px" />
      <select id="type">
        <option value="">any type</option>
        <option>episodic</option><option>semantic</option><option>procedural</option>
        <option>antipattern</option><option>reasoning</option><option>correction</option>
      </select>
      <input id="agent" placeholder="agent uuid (optional)" style="width:200px" />
      <button class="act" onclick="loadMem(0)">Load</button>
      <span id="memmsg" class="sub"></span>
    </div>
    <table><thead><tr><th>Content</th><th>Type</th><th>Agent</th><th>Created</th><th></th></tr></thead>
    <tbody id="rows"></tbody></table>
    <div class="row" style="margin-top:14px">
      <button class="act" id="prev" onclick="pageBy(-1)">Prev</button>
      <button class="act" id="next" onclick="pageBy(1)">Next</button>
      <span id="pager" class="sub"></span>
    </div>
  </section>
  <section id="query" class="hidden">
    <div class="row">
      <input id="qkey" type="password" placeholder="admin key (x-api-key)" style="width:220px" />
      <button class="act" onclick="runMql()">Run</button>
      <span id="qmsg" class="sub"></span>
    </div>
    <textarea id="mql" spellcheck="false"
      style="width:100%;height:110px;margin-top:10px;box-sizing:border-box;background:#0b0b0d;color:#e4e4e7;border:1px solid #1f1f23;border-radius:8px;padding:10px;font-family:ui-monospace,SFMono-Regular,monospace;font-size:13px"
      placeholder="RECALL WHERE memory_type = &quot;semantic&quot; LIMIT 20"></textarea>
    <p class="muted" style="margin-top:6px">Runs raw MQL through the engine. Read only browsing, no context assembly. Results ordered by score.</p>
    <table><thead><tr><th>Content</th><th>Type</th><th>Agent</th><th>Score</th></tr></thead>
    <tbody id="qrows"></tbody></table>
  </section>
</main>
<script>
function showTab(t) {
  for (const n of ["health","memories","query"]) {
    document.getElementById(n).classList.toggle("hidden", n !== t);
    document.getElementById("tab-"+n).classList.toggle("active", n === t);
  }
}
// ---- Health ----
function parse(text) {
  const m = {};
  for (const line of text.split("\n")) {
    if (!line || line[0] === "#") continue;
    const sp = line.lastIndexOf(" ");
    if (sp < 0) continue;
    const key = line.slice(0, sp).trim();
    const val = parseFloat(line.slice(sp + 1));
    if (!isFinite(val)) continue;
    const name = key.indexOf("{") >= 0 ? key.slice(0, key.indexOf("{")) : key;
    m[name] = (m[name] || 0) + val;
  }
  return m;
}
function fmtDur(s){s=Math.floor(s);const d=Math.floor(s/86400);s%=86400;const h=Math.floor(s/3600);s%=3600;const mi=Math.floor(s/60);return (d?d+"d ":"")+(h?h+"h ":"")+mi+"m";}
function fmtBytes(b){if(!b)return "-";const u=["B","KB","MB","GB","TB"];let i=0;while(b>=1024&&i<u.length-1){b/=1024;i++;}return b.toFixed(1)+" "+u[i];}
let prev=null, prevT=0;
function card(l,v,c){return `<div class="card"><div class="label">${l}</div><div class="value ${c||""}">${v}</div></div>`;}
async function tick() {
  let m, ok = true;
  try { m = parse(await (await fetch("/metrics",{cache:"no-store"})).text()); } catch { ok=false; m={}; }
  const up = ok && m["mentedb_up"] === 1;
  document.getElementById("dot").className = "dot " + (up?"up":"down");
  document.getElementById("uptime").textContent = up ? "up "+fmtDur(m["mentedb_uptime_seconds"]||0) : "unreachable";
  const now = Date.now()/1000; let reqRate=null, cpu=null;
  if (prev && now>prevT){const dt=now-prevT;
    if(m["mentedb_http_requests_total"]!=null&&prev["mentedb_http_requests_total"]!=null) reqRate=Math.max(0,(m["mentedb_http_requests_total"]-prev["mentedb_http_requests_total"])/dt);
    if(m["process_cpu_seconds_total"]!=null&&prev["process_cpu_seconds_total"]!=null) cpu=Math.max(0,(m["process_cpu_seconds_total"]-prev["process_cpu_seconds_total"])/dt);
  }
  prev=m; prevT=now;
  document.getElementById("grid").innerHTML = [
    card("Status", up?"Up":"Down", up?"ok":""),
    card("Memories stored",(m["mentedb_memory_count"]||0).toLocaleString()),
    card("Live cluster nodes", m["mentedb_cluster_live_nodes"]!=null?m["mentedb_cluster_live_nodes"]:"-"),
    card("Requests / sec", reqRate==null?"…":reqRate.toFixed(1)),
    card("CPU (cores)", cpu==null?(isFinite(m["process_cpu_seconds_total"])?"…":"n/a"):cpu.toFixed(2)),
    card("Resident memory", isFinite(m["process_resident_memory_bytes"])?fmtBytes(m["process_resident_memory_bytes"]):"n/a"),
  ].join("");
}
tick(); setInterval(tick, 2000);
// ---- Memories ----
let offset = 0, limit = 50, total = 0;
function esc(s){return (s||"").replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));}
function key(){ return document.getElementById("key").value.trim(); }
async function loadMem(off) {
  const k = key();
  if (!k) { document.getElementById("memmsg").textContent = "enter the admin key"; return; }
  sessionStorage.setItem("mdb_key", k);
  offset = off;
  const p = new URLSearchParams({ limit, offset });
  const q = document.getElementById("q").value.trim(); if (q) p.set("q", q);
  const t = document.getElementById("type").value; if (t) p.set("type", t);
  const a = document.getElementById("agent").value.trim(); if (a) p.set("agent", a);
  document.getElementById("memmsg").textContent = "loading…";
  let r;
  try { r = await fetch("/v1/admin/memories?"+p, { headers: { "x-api-key": k } }); }
  catch { document.getElementById("memmsg").textContent = "request failed"; return; }
  if (!r.ok) { document.getElementById("memmsg").textContent = "error "+r.status+" (check the admin key)"; return; }
  const d = await r.json();
  total = d.total;
  document.getElementById("rows").innerHTML = (d.memories||[]).map(mem => {
    const dt = mem.created_at ? new Date(mem.created_at/1000).toISOString().slice(0,16).replace("T"," ") : "";
    const ag = (mem.agent_id||"").slice(0,8);
    return `<tr><td class="content">${esc(mem.content)}</td><td><span class="pill">${esc(mem.memory_type)}</span></td><td class="mono">${ag}</td><td class="mono">${dt}</td><td><button class="del" onclick="del('${mem.id}')">delete</button></td></tr>`;
  }).join("");
  document.getElementById("memmsg").textContent = "";
  document.getElementById("pager").textContent = `${offset+1}–${Math.min(offset+limit,total)} of ${total.toLocaleString()}`;
  document.getElementById("prev").disabled = offset <= 0;
  document.getElementById("next").disabled = offset+limit >= total;
}
function pageBy(dir){ const n = offset + dir*limit; if (n>=0 && n<total) loadMem(n); }
async function del(id) {
  if (!confirm("Delete this memory?")) return;
  const r = await fetch("/v1/admin/memories/"+id, { method:"DELETE", headers: { "x-api-key": key() } });
  if (r.ok) loadMem(offset); else alert("delete failed: "+r.status);
}
// ---- Query ----
async function runMql() {
  const k = document.getElementById("qkey").value.trim();
  if (!k) { document.getElementById("qmsg").textContent = "enter the admin key"; return; }
  sessionStorage.setItem("mdb_key", k);
  const mql = document.getElementById("mql").value.trim();
  if (!mql) { document.getElementById("qmsg").textContent = "enter a query"; return; }
  document.getElementById("qmsg").textContent = "running…";
  let r;
  try {
    r = await fetch("/v1/admin/mql", { method:"POST",
      headers: { "x-api-key": k, "content-type": "application/json" },
      body: JSON.stringify({ mql }) });
  } catch { document.getElementById("qmsg").textContent = "request failed"; return; }
  if (!r.ok) {
    const e = await r.json().catch(()=>null);
    document.getElementById("qmsg").textContent = (e && e.error) ? e.error : ("error "+r.status);
    return;
  }
  const d = await r.json();
  document.getElementById("qrows").innerHTML = (d.memories||[]).map(mem => {
    const ag = (mem.agent_id||"").slice(0,8);
    const sc = typeof mem.score === "number" ? mem.score.toFixed(3) : "";
    return `<tr><td class="content">${esc(mem.content)}</td><td><span class="pill">${esc(mem.memory_type)}</span></td><td class="mono">${ag}</td><td class="mono">${sc}</td></tr>`;
  }).join("");
  document.getElementById("qmsg").textContent = (d.count||0)+" result"+((d.count===1)?"":"s");
}
window.addEventListener("load", () => {
  const s = sessionStorage.getItem("mdb_key");
  if (s) { document.getElementById("key").value = s; document.getElementById("qkey").value = s; }
});
</script>
</body>
</html>
"##;

/// `GET /console`: the bundled management console.
pub async fn handler() -> Html<&'static str> {
    Html(PAGE)
}
