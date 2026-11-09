//! The memory tool backend end to end on a real database: every command,
//! per scope isolation, edits that must never be dropped as duplicates, and
//! durability across a crash.

use mentedb::MenteDb;
use mentedb::memory_tool::{
    self, MemoryToolOutcome, MemoryToolRequest, MemoryToolResponse, MemoryToolScope, SOURCE_TAG,
};
use mentedb_core::types::{AgentId, UserId};
use mentedb_embedding::HashEmbeddingProvider;
use serde_json::{Value, json};

fn open(dir: &std::path::Path) -> MenteDb {
    MenteDb::open_with_embedder(dir, Box::new(HashEmbeddingProvider::new(64))).unwrap()
}

fn run(db: &MenteDb, scope: &MemoryToolScope, input: Value) -> MemoryToolOutcome {
    let req: MemoryToolRequest = serde_json::from_value(input).unwrap();
    memory_tool::handle(db, scope, &req).unwrap()
}

fn ok(db: &MenteDb, scope: &MemoryToolScope, input: Value) -> String {
    let out = run(db, scope, input.clone());
    assert!(
        !out.response.is_error,
        "{input} failed: {}",
        out.response.content
    );
    out.response.content
}

fn err(db: &MenteDb, scope: &MemoryToolScope, input: Value) -> String {
    let out = run(db, scope, input.clone());
    assert!(
        out.response.is_error,
        "{input} should fail: {}",
        out.response.content
    );
    assert_eq!(out.memory_delta, 0);
    out.response.content
}

fn view(db: &MenteDb, scope: &MemoryToolScope, path: &str) -> MemoryToolResponse {
    run(db, scope, json!({"command": "view", "path": path})).response
}

fn scope(agent: AgentId) -> MemoryToolScope {
    MemoryToolScope::new(agent, UserId::nil())
}

#[test]
fn every_command_round_trips() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    let s = scope(AgentId::new());

    // Empty store: the root lists, nothing else exists.
    assert_eq!(
        ok(&db, &s, json!({"command": "view", "path": "/memories"})),
        "Here're the files and directories up to 2 levels deep in /memories, excluding hidden items and node_modules:\n0B\t/memories"
    );
    assert_eq!(
        err(
            &db,
            &s,
            json!({"command": "view", "path": "/memories/none.md"})
        ),
        "The path /memories/none.md does not exist. Please provide a valid path."
    );

    // create, then view the file and the listing.
    let out = run(
        &db,
        &s,
        json!({"command": "create", "path": "/memories/prefs.md", "file_text": "color: blue\nfood: rice\n"}),
    );
    assert_eq!(
        out.response,
        MemoryToolResponse::ok("File created successfully at: /memories/prefs.md")
    );
    assert_eq!(out.memory_delta, 1);
    assert_eq!(
        ok(
            &db,
            &s,
            json!({"command": "view", "path": "/memories/prefs.md"})
        ),
        "Here's the content of /memories/prefs.md with line numbers:\n     1\tcolor: blue\n     2\tfood: rice"
    );
    assert_eq!(
        ok(
            &db,
            &s,
            json!({"command": "view", "path": "/memories/prefs.md", "view_range": [2, -1]})
        ),
        "Here's the content of /memories/prefs.md with line numbers:\n     2\tfood: rice"
    );
    assert!(
        ok(&db, &s, json!({"command": "view", "path": "/memories"}))
            .ends_with("\n23B\t/memories\n23B\t/memories/prefs.md")
    );

    // create over an existing file overwrites it in place.
    let out = run(
        &db,
        &s,
        json!({"command": "create", "path": "/memories/prefs.md", "file_text": "color: blue\nfood: rice\nsport: tennis"}),
    );
    assert_eq!(out.memory_delta, 0);
    assert_eq!(memory_tool::list_files(&db, &s).len(), 1);

    // str_replace: one match edits, zero or many refuse.
    assert_eq!(
        ok(
            &db,
            &s,
            json!({"command": "str_replace", "path": "/memories/prefs.md", "old_str": "blue", "new_str": "green"})
        ),
        "The memory file has been edited."
    );
    assert!(
        err(
            &db,
            &s,
            json!({"command": "str_replace", "path": "/memories/prefs.md", "old_str": "purple", "new_str": "x"})
        )
        .contains("did not appear verbatim")
    );
    assert!(
        err(
            &db,
            &s,
            json!({"command": "str_replace", "path": "/memories/prefs.md", "old_str": ": ", "new_str": "="})
        )
        .contains("Multiple occurrences")
    );
    assert!(
        err(
            &db,
            &s,
            json!({"command": "str_replace", "path": "/memories/gone.md", "old_str": "a"})
        )
        .contains("does not exist")
    );

    // insert at the top and past the end.
    assert_eq!(
        ok(
            &db,
            &s,
            json!({"command": "insert", "path": "/memories/prefs.md", "insert_line": 0, "insert_text": "# prefs\n"})
        ),
        "The file /memories/prefs.md has been edited."
    );
    assert!(
        err(
            &db,
            &s,
            json!({"command": "insert", "path": "/memories/prefs.md", "insert_line": 99, "insert_text": "x"})
        )
        .contains("Invalid `insert_line`")
    );
    let file = memory_tool::find_file(&db, &s, "/memories/prefs.md").unwrap();
    assert_eq!(
        file.node.content,
        "# prefs\ncolor: green\nfood: rice\nsport: tennis"
    );
    assert!(
        !file.node.embedding.is_empty(),
        "edited file is re-embedded"
    );
    assert!(file.node.tags.iter().any(|t| t == SOURCE_TAG));

    // rename a file, then a directory.
    assert_eq!(
        ok(
            &db,
            &s,
            json!({"command": "rename", "old_path": "/memories/prefs.md", "new_path": "/memories/me/prefs.md"})
        ),
        "Successfully renamed /memories/prefs.md to /memories/me/prefs.md"
    );
    ok(
        &db,
        &s,
        json!({"command": "create", "path": "/memories/me/todo.md", "file_text": "- ship it"}),
    );
    ok(
        &db,
        &s,
        json!({"command": "rename", "old_path": "/memories/me", "new_path": "/memories/user"}),
    );
    assert!(memory_tool::find_file(&db, &s, "/memories/me/prefs.md").is_none());
    assert!(memory_tool::find_file(&db, &s, "/memories/user/prefs.md").is_some());
    assert!(memory_tool::find_file(&db, &s, "/memories/user/todo.md").is_some());
    assert!(!view(&db, &s, "/memories/user").is_error);
    assert!(view(&db, &s, "/memories/me").is_error);
    // The old path tag left the index with the rename.
    assert!(
        db.memory_ids_with_tag(&memory_tool::path_tag("/memories/me/prefs.md"))
            .is_empty()
    );

    // A destination taken by a file or a directory is refused.
    ok(
        &db,
        &s,
        json!({"command": "create", "path": "/memories/other.md", "file_text": "x"}),
    );
    assert!(
        err(
            &db,
            &s,
            json!({"command": "rename", "old_path": "/memories/other.md", "new_path": "/memories/user/todo.md"})
        )
        .contains("already exists")
    );
    assert!(
        err(
            &db,
            &s,
            json!({"command": "rename", "old_path": "/memories/other.md", "new_path": "/memories/user"})
        )
        .contains("already exists")
    );
    assert!(
        err(
            &db,
            &s,
            json!({"command": "rename", "old_path": "/memories/nope", "new_path": "/memories/x"})
        )
        .contains("does not exist")
    );

    // delete a file, then a directory with everything under it.
    let out = run(
        &db,
        &s,
        json!({"command": "delete", "path": "/memories/other.md"}),
    );
    assert_eq!(
        out.response,
        MemoryToolResponse::ok("Successfully deleted /memories/other.md")
    );
    assert_eq!(out.memory_delta, -1);
    let out = run(
        &db,
        &s,
        json!({"command": "delete", "path": "/memories/user"}),
    );
    assert_eq!(out.memory_delta, -2);
    assert!(memory_tool::list_files(&db, &s).is_empty());
    assert_eq!(db.memory_count(), 0);
    assert!(
        err(
            &db,
            &s,
            json!({"command": "delete", "path": "/memories/user"})
        )
        .contains("does not exist")
    );

    // The root itself can be neither created, deleted nor renamed.
    for input in [
        json!({"command": "create", "path": "/memories", "file_text": "x"}),
        json!({"command": "delete", "path": "/memories"}),
        json!({"command": "rename", "old_path": "/memories", "new_path": "/memories/x"}),
    ] {
        assert!(err(&db, &s, input).contains("directory itself"));
    }
}

#[test]
fn traversal_is_refused_before_touching_the_store() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    let s = scope(AgentId::new());
    for path in [
        "/memories/../etc/passwd",
        "/memories/%2e%2e/x",
        "/memories/%252e%252e/x",
        "/memories/..\\x",
        "/etc/passwd",
        "memories/x",
    ] {
        let msg = err(
            &db,
            &s,
            json!({"command": "create", "path": path, "file_text": "pwned"}),
        );
        assert!(msg.contains("Invalid path"), "{path}: {msg}");
        assert!(
            err(
                &db,
                &s,
                json!({"command": "rename", "old_path": "/memories/a", "new_path": path})
            )
            .contains("Invalid path")
        );
    }
    assert_eq!(db.memory_count(), 0);
}

#[test]
fn scopes_never_see_each_other() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    let alice = scope(AgentId::new());
    let bob = scope(AgentId::new());
    let alice_for_user = MemoryToolScope::new(alice.agent_id, UserId::new());

    ok(
        &db,
        &alice,
        json!({"command": "create", "path": "/memories/secret.md", "file_text": "alice only"}),
    );

    // Bob sees an empty root and no file at the same path.
    assert!(
        view(&db, &bob, "/memories")
            .content
            .ends_with("\n0B\t/memories")
    );
    assert!(view(&db, &bob, "/memories/secret.md").is_error);
    assert!(view(&db, &alice_for_user, "/memories/secret.md").is_error);
    assert!(view(&db, &MemoryToolScope::unowned(), "/memories/secret.md").is_error);
    for input in [
        json!({"command": "str_replace", "path": "/memories/secret.md", "old_str": "alice", "new_str": "bob"}),
        json!({"command": "insert", "path": "/memories/secret.md", "insert_line": 0, "insert_text": "bob"}),
        json!({"command": "delete", "path": "/memories/secret.md"}),
        json!({"command": "rename", "old_path": "/memories/secret.md", "new_path": "/memories/mine.md"}),
    ] {
        let out = run(&db, &bob, input.clone());
        assert!(out.response.is_error, "{input}: {}", out.response.content);
    }

    // Bob writing the same path makes his own file; Alice's is untouched.
    ok(
        &db,
        &bob,
        json!({"command": "create", "path": "/memories/secret.md", "file_text": "bob's"}),
    );
    ok(
        &db,
        &bob,
        json!({"command": "delete", "path": "/memories/secret.md"}),
    );
    assert_eq!(
        memory_tool::find_file(&db, &alice, "/memories/secret.md")
            .unwrap()
            .node
            .content,
        "alice only"
    );
    // Renaming onto a path another scope uses is not a collision either.
    ok(
        &db,
        &bob,
        json!({"command": "create", "path": "/memories/b.md", "file_text": "b"}),
    );
    ok(
        &db,
        &bob,
        json!({"command": "rename", "old_path": "/memories/b.md", "new_path": "/memories/secret.md"}),
    );
    assert_eq!(memory_tool::list_files(&db, &alice).len(), 1);
    assert_eq!(memory_tool::list_files(&db, &bob).len(), 1);
    let file = memory_tool::find_file(&db, &alice, "/memories/secret.md").unwrap();
    assert_eq!(file.node.agent_id, alice.agent_id);
    assert_eq!(file.node.content, "alice only");
}

#[test]
fn identical_files_and_edits_are_never_dropped_as_duplicates() {
    let dir = tempfile::tempdir().unwrap();
    let db = open(dir.path());
    let s = scope(AgentId::new());
    let text = "The user prefers concise answers written in plain words";
    ok(
        &db,
        &s,
        json!({"command": "create", "path": "/memories/a.md", "file_text": text}),
    );
    ok(
        &db,
        &s,
        json!({"command": "create", "path": "/memories/b.md", "file_text": text}),
    );
    assert_eq!(memory_tool::list_files(&db, &s).len(), 2);
    // Edit b to become a near copy of a: the edit must land.
    ok(
        &db,
        &s,
        json!({"command": "create", "path": "/memories/b.md", "file_text": "The user prefers concise answers written in plain words!"}),
    );
    assert!(
        memory_tool::find_file(&db, &s, "/memories/b.md")
            .unwrap()
            .node
            .content
            .ends_with('!')
    );
}

#[test]
fn files_survive_a_crash_and_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let s = scope(AgentId::new());
    {
        let db = open(dir.path());
        for i in 0..20 {
            ok(
                &db,
                &s,
                json!({"command": "create", "path": format!("/memories/notes/f{i}.md"), "file_text": format!("note {i} body")}),
            );
        }
        ok(
            &db,
            &s,
            json!({"command": "str_replace", "path": "/memories/notes/f0.md", "old_str": "body", "new_str": "text"}),
        );
        ok(
            &db,
            &s,
            json!({"command": "rename", "old_path": "/memories/notes/f2.md", "new_path": "/memories/notes/g2.md"}),
        );
        ok(
            &db,
            &s,
            json!({"command": "delete", "path": "/memories/notes/f3.md"}),
        );
        db.simulate_crash();
    }
    let db = open(dir.path());
    let paths: std::collections::HashSet<String> = memory_tool::list_files(&db, &s)
        .into_iter()
        .map(|f| f.path)
        .collect();
    assert_eq!(paths.len(), 19);
    assert!(paths.contains("/memories/notes/g2.md"));
    assert!(!paths.contains("/memories/notes/f2.md"));
    assert!(!paths.contains("/memories/notes/f3.md"));
    assert_eq!(
        memory_tool::find_file(&db, &s, "/memories/notes/f0.md")
            .unwrap()
            .node
            .content,
        "note 0 text"
    );
    assert!(memory_tool::find_file(&db, &s, "/memories/notes/f2.md").is_none());
}
