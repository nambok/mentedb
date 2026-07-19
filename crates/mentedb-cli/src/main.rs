//! `mentedb`: a command-line client for running MQL against MenteDB, either a
//! local data directory (opened in-process) or a running server (over HTTP).

use std::io::{self, BufRead, Write};

use mentedb::MenteDb;
use serde_json::{Value, json};

const USAGE: &str = "\
mentedb - MenteDB command-line client

USAGE:
    mentedb query [connection] [--format table|json|csv] <MQL>
    mentedb repl  [connection] [--format table|json|csv]

CONNECTION (pick one):
    --data-dir <DIR>                 open a local data directory (no server; the
                                     directory must not be open by a server)
    --url <URL> --admin-key <KEY>    connect to a running server

EXAMPLES:
    mentedb query --data-dir ./data 'RECALL memories WHERE type = semantic LIMIT 10'
    mentedb query --url http://localhost:6677 --admin-key \"$KEY\" 'RECALL memories LIMIT 5'
    mentedb repl --data-dir ./data
";

enum Conn {
    Local(String),
    Remote { url: String, key: String },
}

struct Opts {
    conn: Conn,
    format: String,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 || args[1] == "--help" || args[1] == "-h" {
        print!("{USAGE}");
        return;
    }
    let cmd = args[1].clone();

    let (mut data_dir, mut url, mut key) = (None, None, None);
    let mut format = "table".to_string();
    let mut mql_parts: Vec<String> = Vec::new();
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--data-dir" => {
                data_dir = args.get(i + 1).cloned();
                i += 2;
            }
            "--url" => {
                url = args.get(i + 1).cloned();
                i += 2;
            }
            "--admin-key" => {
                key = args.get(i + 1).cloned();
                i += 2;
            }
            "--format" => {
                format = args.get(i + 1).cloned().unwrap_or(format);
                i += 2;
            }
            other => {
                mql_parts.push(other.to_string());
                i += 1;
            }
        }
    }

    let conn = match (data_dir, url, key) {
        (Some(d), _, _) => Conn::Local(d),
        (None, Some(u), Some(k)) => Conn::Remote { url: u, key: k },
        _ => {
            eprintln!("error: specify either --data-dir, or --url with --admin-key\n\n{USAGE}");
            std::process::exit(2);
        }
    };
    let opts = Opts { conn, format };

    match cmd.as_str() {
        "query" => {
            let mql = mql_parts.join(" ");
            if mql.trim().is_empty() {
                eprintln!("error: no MQL query given");
                std::process::exit(2);
            }
            match run_query(&opts.conn, &mql) {
                Ok(rows) => print_results(&rows, &opts.format),
                Err(e) => {
                    eprintln!("error: {e}");
                    std::process::exit(1);
                }
            }
        }
        "repl" => repl(&opts),
        other => {
            eprintln!("error: unknown command '{other}'\n\n{USAGE}");
            std::process::exit(2);
        }
    }
}

fn repl(opts: &Opts) {
    println!("MenteDB CLI. Enter an MQL query, or \\q to quit.");
    let stdin = io::stdin();
    loop {
        print!("mentedb> ");
        io::stdout().flush().ok();
        let mut line = String::new();
        if stdin.lock().read_line(&mut line).unwrap_or(0) == 0 {
            println!();
            break;
        }
        let q = line.trim();
        if q.is_empty() {
            continue;
        }
        if q == "\\q" || q == "\\quit" || q == "exit" {
            break;
        }
        match run_query(&opts.conn, q) {
            Ok(rows) => print_results(&rows, &opts.format),
            Err(e) => eprintln!("error: {e}"),
        }
    }
}

/// Run the query and return the matches as JSON memory objects
/// (content, memory_type, agent_id, created_at, score).
fn run_query(conn: &Conn, mql: &str) -> Result<Vec<Value>, String> {
    match conn {
        Conn::Local(dir) => {
            let db = MenteDb::open(std::path::Path::new(dir)).map_err(|e| e.to_string())?;
            let results = db.query(mql).map_err(|e| e.to_string())?;
            Ok(results
                .iter()
                .map(|s| {
                    json!({
                        "id": s.memory.id.to_string(),
                        "memory_type": format!("{:?}", s.memory.memory_type).to_lowercase(),
                        "agent_id": s.memory.agent_id.to_string(),
                        "content": s.memory.content,
                        "created_at": s.memory.created_at,
                        "score": s.score,
                    })
                })
                .collect())
        }
        Conn::Remote { url, key } => {
            let endpoint = format!("{}/v1/admin/mql", url.trim_end_matches('/'));
            let resp = reqwest::blocking::Client::new()
                .post(&endpoint)
                .header("x-api-key", key)
                .json(&json!({ "mql": mql }))
                .send()
                .map_err(|e| e.to_string())?;
            if !resp.status().is_success() {
                return Err(format!("server returned {}", resp.status()));
            }
            let body: Value = resp.json().map_err(|e| e.to_string())?;
            Ok(body
                .get("memories")
                .and_then(|m| m.as_array())
                .cloned()
                .unwrap_or_default())
        }
    }
}

fn field(v: &Value, k: &str) -> String {
    match v.get(k) {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Number(n)) => n.to_string(),
        Some(other) => other.to_string(),
        None => String::new(),
    }
}

fn print_results(rows: &[Value], format: &str) {
    match format {
        "json" => println!(
            "{}",
            serde_json::to_string_pretty(&rows).unwrap_or_else(|_| "[]".into())
        ),
        "csv" => {
            println!("score,type,agent,content");
            for r in rows {
                println!(
                    "{},{},{},{}",
                    field(r, "score"),
                    field(r, "memory_type"),
                    field(r, "agent_id"),
                    csv_escape(&field(r, "content")),
                );
            }
        }
        _ => print_table(rows),
    }
}

fn print_table(rows: &[Value]) {
    if rows.is_empty() {
        println!("(0 rows)");
        return;
    }
    println!("{:>7}  {:<11}  {:<8}  CONTENT", "SCORE", "TYPE", "AGENT");
    for r in rows {
        let score = field(r, "score");
        let score = score
            .parse::<f64>()
            .map(|f| format!("{f:.3}"))
            .unwrap_or(score);
        let mtype = field(r, "memory_type");
        let agent: String = field(r, "agent_id").chars().take(8).collect();
        let content = {
            let c = field(r, "content").replace('\n', " ");
            if c.chars().count() > 70 {
                format!("{}…", c.chars().take(69).collect::<String>())
            } else {
                c
            }
        };
        println!("{score:>7}  {mtype:<11}  {agent:<8}  {content}");
    }
    println!(
        "({} row{})",
        rows.len(),
        if rows.len() == 1 { "" } else { "s" }
    );
}

fn csv_escape(s: &str) -> String {
    if s.contains([',', '"', '\n']) {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}
