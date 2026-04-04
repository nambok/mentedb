import type {
  HealthResponse,
  StatsResponse,
  MemoryNode,
  CreateMemoryRequest,
  CreateMemoryResponse,
  CreateEdgeRequest,
  SearchRequest,
  SearchResponse,
  RecallRequest,
  RecallResponse,
} from "./types";

const DEFAULT_BASE_URL = "/v1";

class ApiClient {
  private baseUrl: string;
  private token: string | null = null;

  constructor(baseUrl: string = DEFAULT_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  configure(baseUrl: string, token?: string) {
    this.baseUrl = baseUrl;
    this.token = token ?? null;
  }

  setToken(token: string) {
    this.token = token;
  }

  private async request<T>(
    path: string,
    options: RequestInit = {}
  ): Promise<T> {
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
      ...(options.headers as Record<string, string>),
    };

    if (this.token) {
      headers["Authorization"] = `Bearer ${this.token}`;
    }

    const response = await fetch(`${this.baseUrl}${path}`, {
      ...options,
      headers,
    });

    if (!response.ok) {
      const body = await response.json().catch(() => ({ error: response.statusText }));
      throw new ApiError(response.status, body.error || "Unknown error");
    }

    return response.json();
  }

  async health(): Promise<HealthResponse> {
    return this.request<HealthResponse>("/health");
  }

  async stats(): Promise<StatsResponse> {
    return this.request<StatsResponse>("/stats");
  }

  async getMemory(id: string): Promise<MemoryNode> {
    return this.request<MemoryNode>(`/memories/${id}`);
  }

  async createMemory(data: CreateMemoryRequest): Promise<CreateMemoryResponse> {
    return this.request<CreateMemoryResponse>("/memories", {
      method: "POST",
      body: JSON.stringify(data),
    });
  }

  async deleteMemory(id: string): Promise<{ status: string }> {
    return this.request<{ status: string }>(`/memories/${id}`, {
      method: "DELETE",
    });
  }

  async search(data: SearchRequest): Promise<SearchResponse> {
    return this.request<SearchResponse>("/search", {
      method: "POST",
      body: JSON.stringify(data),
    });
  }

  async recall(data: RecallRequest): Promise<RecallResponse> {
    return this.request<RecallResponse>("/recall", {
      method: "POST",
      body: JSON.stringify(data),
    });
  }

  async createEdge(data: CreateEdgeRequest): Promise<{ status: string }> {
    return this.request<{ status: string }>("/edges", {
      method: "POST",
      body: JSON.stringify(data),
    });
  }

  createWebSocket(): WebSocket {
    const wsUrl = this.baseUrl.replace(/^http/, "ws") + "/ws/stream";
    return new WebSocket(wsUrl);
  }
}

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string
  ) {
    super(message);
    this.name = "ApiError";
  }
}

export const api = new ApiClient();
export default api;
