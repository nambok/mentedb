import { spawn, type ChildProcess } from 'child_process';
import * as fs from 'fs';
import * as net from 'net';
import * as os from 'os';
import * as path from 'path';

import { MemoryTool, MemoryToolError, MEMORY_TOOL_DEFINITION } from '../npm/memoryTool';

type Captured = { url: string; init: RequestInit };

function mockFetch(status: number, reply: unknown, captured: Captured[]): typeof fetch {
  return (async (url: string | URL | Request, init?: RequestInit) => {
    captured.push({ url: String(url), init: init ?? {} });
    return new Response(typeof reply === 'string' ? reply : JSON.stringify(reply), { status });
  }) as typeof fetch;
}

describe('MemoryTool', () => {
  it('exposes the memory tool definition', () => {
    expect(MEMORY_TOOL_DEFINITION).toEqual({ type: 'memory_20250818', name: 'memory' });
    expect(MemoryTool.TOOL).toBe(MEMORY_TOOL_DEFINITION);
  });

  it('posts the tool input with the scope and token', async () => {
    const captured: Captured[] = [];
    const tool = new MemoryTool({
      baseUrl: 'http://localhost:6677/',
      token: 'jwt-token',
      agentId: 'agent-1',
      userId: 'user-1',
      fetch: mockFetch(200, { content: 'File created successfully at: /memories/a.md', is_error: false }, captured),
    });
    const out = await tool.execute({ command: 'create', path: '/memories/a.md', file_text: 'hi' });
    expect(out).toEqual({ content: 'File created successfully at: /memories/a.md', is_error: false });
    expect(captured[0].url).toBe('http://localhost:6677/v1/memory_tool');
    expect(captured[0].init.method).toBe('POST');
    const headers = captured[0].init.headers as Record<string, string>;
    expect(headers.authorization).toBe('Bearer jwt-token');
    expect(JSON.parse(captured[0].init.body as string)).toEqual({
      command: 'create',
      path: '/memories/a.md',
      file_text: 'hi',
      agent_id: 'agent-1',
      user_id: 'user-1',
    });
  });

  it('targets a named database and sends no token when none is set', async () => {
    const captured: Captured[] = [];
    const tool = new MemoryTool({
      database: 'team a',
      fetch: mockFetch(200, { content: 'ok', is_error: false }, captured),
    });
    await tool.execute({ command: 'view', path: '/memories' });
    expect(captured[0].url).toBe('http://localhost:6677/v1/db/team%20a/memory_tool');
    expect((captured[0].init.headers as Record<string, string>).authorization).toBeUndefined();
  });

  it('turns memory tool_use blocks into tool_result blocks', async () => {
    const captured: Captured[] = [];
    const tool = new MemoryTool({
      fetch: mockFetch(
        200,
        { content: 'The path /memories/x does not exist. Please provide a valid path.', is_error: true },
        captured,
      ),
    });
    const blocks = await tool.results([
      { type: 'text' },
      { type: 'tool_use', id: 'toolu_1', name: 'memory', input: { command: 'view', path: '/memories/x' } },
      { type: 'tool_use', id: 'toolu_2', name: 'other_tool', input: {} },
    ]);
    expect(captured).toHaveLength(1);
    expect(blocks).toEqual([
      {
        type: 'tool_result',
        tool_use_id: 'toolu_1',
        content: 'The path /memories/x does not exist. Please provide a valid path.',
        is_error: true,
      },
    ]);
  });

  it('leaves is_error off a successful result', async () => {
    const tool = new MemoryTool({
      fetch: mockFetch(200, { content: 'Successfully deleted /memories/a.md', is_error: false }, []),
    });
    const block = await tool.handle({
      type: 'tool_use',
      id: 'toolu_9',
      name: 'memory',
      input: { command: 'delete', path: '/memories/a.md' },
    });
    expect(block).toEqual({ type: 'tool_result', tool_use_id: 'toolu_9', content: 'Successfully deleted /memories/a.md' });
  });

  it('throws MemoryToolError on HTTP errors', async () => {
    const tool = new MemoryTool({ fetch: mockFetch(403, { error: 'token is read-only' }, []) });
    await expect(tool.execute({ command: 'create', path: '/memories/a', file_text: '' })).rejects.toMatchObject({
      name: 'MemoryToolError',
      status: 403,
    });
    await expect(tool.execute({ command: 'view', path: '/memories' })).rejects.toBeInstanceOf(MemoryToolError);
  });
});

// ---- real server round trip ------------------------------------------------

const REPO = path.resolve(__dirname, '..', '..', '..');

function serverBinary(): string | undefined {
  if (process.env.MENTEDB_SERVER_BIN) return process.env.MENTEDB_SERVER_BIN;
  for (const profile of ['debug', 'release']) {
    const candidate = path.join(REPO, 'target', profile, 'mentedb-server');
    if (fs.existsSync(candidate)) return candidate;
  }
  return undefined;
}

function freePort(): Promise<number> {
  return new Promise((resolve, reject) => {
    const srv = net.createServer();
    srv.once('error', reject);
    srv.listen(0, '127.0.0.1', () => {
      const port = (srv.address() as net.AddressInfo).port;
      srv.close(() => resolve(port));
    });
  });
}

const binary = serverBinary();
const describeServer = binary ? describe : describe.skip;

describeServer('MemoryTool against a real mentedb-server', () => {
  let proc: ChildProcess;
  let url: string;
  let dataDir: string;

  beforeAll(async () => {
    dataDir = fs.mkdtempSync(path.join(os.tmpdir(), 'mentedb-memtool-'));
    const port = await freePort();
    const grpcPort = await freePort();
    proc = spawn(binary as string, ['--data-dir', dataDir, '--port', String(port), '--grpc-port', String(grpcPort)], {
      stdio: 'ignore',
    });
    url = `http://127.0.0.1:${port}`;
    const deadline = Date.now() + 30_000;
    while (Date.now() < deadline) {
      try {
        if ((await fetch(`${url}/v1/health`)).ok) return;
      } catch {
        // not up yet
      }
      await new Promise((r) => setTimeout(r, 100));
    }
    throw new Error('mentedb-server did not start');
  }, 40_000);

  afterAll(async () => {
    if (proc && proc.exitCode === null) {
      const exited = new Promise((r) => proc.once('exit', r));
      proc.kill('SIGTERM');
      await exited;
    }
    fs.rmSync(dataDir, { recursive: true, force: true });
  }, 20_000);

  it('runs every command and keeps agents apart', async () => {
    const alice = new MemoryTool({ baseUrl: url, agentId: '00000000-0000-0000-0000-000000000001' });
    const bob = new MemoryTool({ baseUrl: url, agentId: '00000000-0000-0000-0000-000000000002' });

    expect(await alice.execute({ command: 'create', path: '/memories/notes.md', file_text: 'alpha\nbeta\n' })).toEqual({
      content: 'File created successfully at: /memories/notes.md',
      is_error: false,
    });
    expect((await alice.execute({ command: 'view', path: '/memories/notes.md' })).content).toBe(
      "Here's the content of /memories/notes.md with line numbers:\n     1\talpha\n     2\tbeta",
    );
    expect(
      (await alice.execute({ command: 'str_replace', path: '/memories/notes.md', old_str: 'beta', new_str: 'gamma' }))
        .is_error,
    ).toBe(false);
    expect(
      (await alice.execute({ command: 'insert', path: '/memories/notes.md', insert_line: 0, insert_text: 'top' }))
        .is_error,
    ).toBe(false);
    expect(
      (await alice.execute({ command: 'rename', old_path: '/memories/notes.md', new_path: '/memories/done/notes.md' }))
        .is_error,
    ).toBe(false);
    expect(
      (await alice.execute({ command: 'view', path: '/memories/done/notes.md', view_range: [1, -1] })).content,
    ).toMatch(/ {5}1\ttop\n {5}2\talpha\n {5}3\tgamma$/);

    expect((await bob.execute({ command: 'view', path: '/memories' })).content).toMatch(/\n0B\t\/memories$/);
    expect((await bob.execute({ command: 'view', path: '/memories/done/notes.md' })).is_error).toBe(true);

    const bad = await alice.execute({ command: 'create', path: '/memories/../escape', file_text: 'x' });
    expect(bad.is_error).toBe(true);
    expect(bad.content).toMatch(/^Error: Invalid path/);

    const blocks = await alice.results([
      { type: 'tool_use', id: 'toolu_1', name: 'memory', input: { command: 'delete', path: '/memories/done' } },
    ]);
    expect(blocks).toEqual([
      { type: 'tool_result', tool_use_id: 'toolu_1', content: 'Successfully deleted /memories/done' },
    ]);
    expect((await alice.execute({ command: 'view', path: '/memories' })).content).toMatch(/\n0B\t\/memories$/);
  });
});
