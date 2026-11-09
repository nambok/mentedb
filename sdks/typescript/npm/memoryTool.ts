/**
 * Backend for the client side memory tool (`memory_20250818`), stored in MenteDB.
 *
 * The model sends file style commands (`view`, `create`, `str_replace`,
 * `insert`, `delete`, `rename`) against a `/memories` directory. Your tool
 * loop forwards each `tool_use` input to `POST /v1/memory_tool` and copies the
 * reply into a `tool_result` block. `MemoryTool` does both steps. A
 * self-hosted `mentedb-server` and MenteDB Cloud serve the same contract, so
 * only the base URL and credential change.
 *
 * Uses the global `fetch` (Node 18+) and loads no native code, so it can be
 * imported on its own: `import { MemoryTool } from 'mentedb/npm/memoryTool'`.
 */

/** Default address of a self-hosted `mentedb-server`. */
export const DEFAULT_SERVER_URL = 'http://localhost:6677';

/** Tool definition to pass in `tools` on a Messages API request. */
export const MEMORY_TOOL_DEFINITION = { type: 'memory_20250818', name: 'memory' } as const;

/** The input of a memory `tool_use` block. */
export interface MemoryToolInput {
  command: 'view' | 'create' | 'str_replace' | 'insert' | 'delete' | 'rename' | string;
  path?: string;
  view_range?: number[];
  file_text?: string;
  old_str?: string;
  new_str?: string;
  insert_line?: number;
  insert_text?: string;
  old_path?: string;
  new_path?: string;
}

/** What the server returns for one command. */
export interface MemoryToolResult {
  content: string;
  is_error: boolean;
}

/** A `tool_result` content block, ready to send back in a user message. */
export interface MemoryToolResultBlock {
  type: 'tool_result';
  tool_use_id: string;
  content: string;
  is_error?: boolean;
}

/** The parts of an assistant content block this helper reads. */
export interface ContentBlockLike {
  type: string;
  id?: string;
  name?: string;
  input?: unknown;
}

export interface MemoryToolOptions {
  /** e.g. `http://localhost:6677` (self-hosted) or `https://api.mentedb.com` (cloud). */
  baseUrl?: string;
  /** Bearer token: a server JWT, or an `mdb_` key for the cloud. Unset for a server without auth. */
  token?: string;
  /** A named database on a self-hosted server. Unset uses the default database. */
  database?: string;
  /**
   * Whose files to act on. A token bound to an agent always acts as that
   * agent; this is for unbound or admin tokens and servers without auth.
   */
  agentId?: string;
  /** Optionally split one agent's files further by user. */
  userId?: string;
  /** Per request timeout in milliseconds. Default 30000. */
  timeoutMs?: number;
  /** A `fetch` implementation. Defaults to the global one. */
  fetch?: typeof fetch;
}

/** Auth, transport and server failures. Command failures are results, not errors. */
export class MemoryToolError extends Error {
  readonly status?: number;
  constructor(message: string, status?: number) {
    super(message);
    this.name = 'MemoryToolError';
    this.status = status;
  }
}

/** Runs memory tool calls against a MenteDB server. */
export class MemoryTool {
  /** Tool definition to pass in `tools` on a Messages API request. */
  static readonly TOOL = MEMORY_TOOL_DEFINITION;

  readonly url: string;
  private readonly token?: string;
  private readonly agentId?: string;
  private readonly userId?: string;
  private readonly timeoutMs: number;
  private readonly fetchImpl: typeof fetch;

  constructor(options: MemoryToolOptions = {}) {
    const base = (options.baseUrl ?? DEFAULT_SERVER_URL).replace(/\/+$/, '');
    this.url = options.database
      ? `${base}/v1/db/${encodeURIComponent(options.database)}/memory_tool`
      : `${base}/v1/memory_tool`;
    this.token = options.token;
    this.agentId = options.agentId;
    this.userId = options.userId;
    this.timeoutMs = options.timeoutMs ?? 30_000;
    this.fetchImpl = options.fetch ?? fetch;
  }

  /**
   * Run one command. Command failures the model can fix (a missing file, a
   * bad range, an invalid path) come back with `is_error` set; auth and
   * server failures throw `MemoryToolError`.
   */
  async execute(input: MemoryToolInput | Record<string, unknown>): Promise<MemoryToolResult> {
    const body: Record<string, unknown> = {};
    for (const [key, value] of Object.entries(input ?? {})) {
      if (value !== undefined && value !== null) body[key] = value;
    }
    if (this.agentId !== undefined) body.agent_id = this.agentId;
    if (this.userId !== undefined) body.user_id = this.userId;

    const headers: Record<string, string> = { 'content-type': 'application/json' };
    if (this.token) headers.authorization = `Bearer ${this.token}`;

    let response: Response;
    try {
      response = await this.fetchImpl(this.url, {
        method: 'POST',
        headers,
        body: JSON.stringify(body),
        signal: AbortSignal.timeout(this.timeoutMs),
      });
    } catch (err) {
      throw new MemoryToolError(`mentedb unreachable at ${this.url}: ${(err as Error).message}`);
    }
    const text = await response.text();
    if (!response.ok) {
      throw new MemoryToolError(`mentedb ${response.status}: ${text}`, response.status);
    }
    let parsed: unknown;
    try {
      parsed = JSON.parse(text);
    } catch {
      throw new MemoryToolError(`unexpected memory tool reply: ${text}`, response.status);
    }
    const reply = parsed as Partial<MemoryToolResult> | null;
    if (!reply || typeof reply.content !== 'string') {
      throw new MemoryToolError(`unexpected memory tool reply: ${text}`, response.status);
    }
    return { content: reply.content, is_error: Boolean(reply.is_error) };
  }

  /** Turn one memory `tool_use` block into its `tool_result` block. */
  async handle(toolUse: ContentBlockLike): Promise<MemoryToolResultBlock> {
    const result = await this.execute((toolUse.input ?? {}) as Record<string, unknown>);
    const block: MemoryToolResultBlock = {
      type: 'tool_result',
      tool_use_id: toolUse.id ?? '',
      content: result.content,
    };
    if (result.is_error) block.is_error = true;
    return block;
  }

  /**
   * `tool_result` blocks for every memory call in an assistant turn. Pass
   * `response.content`; other blocks are skipped. Send all results in one
   * user message.
   */
  async results(content: ReadonlyArray<ContentBlockLike>): Promise<MemoryToolResultBlock[]> {
    const out: MemoryToolResultBlock[] = [];
    for (const block of content) {
      if (block.type === 'tool_use' && block.name === MEMORY_TOOL_DEFINITION.name) {
        out.push(await this.handle(block));
      }
    }
    return out;
  }
}
