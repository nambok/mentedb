# MenteDB Dashboard

React-based dashboard for visualizing the MenteDB knowledge graph, memories, and cognitive features.

## Stack

- React 19, TypeScript, Vite
- Tailwind CSS v4 (dark theme)
- react-force-graph-2d (knowledge graph visualization)
- lucide-react (icons)

## Getting Started

```bash
cd dashboard
npm install
npm run dev
```

Opens at http://localhost:3000. API requests proxy to the MenteDB server at `localhost:6677`.

## Build

```bash
npm run build
npm run preview
```

## Pages

| Route | Description |
|---|---|
| `/` | Overview -- stats cards, recent activity, memory distribution |
| `/graph` | Interactive force-directed knowledge graph |
| `/memories` | Searchable/filterable memory table with sorting |
| `/cognitive` | Cognition stream, pain signals, phantom memories, trajectory |
| `/spaces` | Per-agent memory spaces and RBAC status |
| `/settings` | Server URL and auth token configuration |

## API Integration

The dashboard connects to the MenteDB REST API (default `http://localhost:6677/v1`). Configure the server URL in Settings.

Endpoints used:
- `GET /v1/health` -- server health
- `GET /v1/stats` -- memory count, uptime
- `POST /v1/memories` -- store memory
- `GET /v1/memories/:id` -- retrieve memory
- `POST /v1/search` -- vector search
- `POST /v1/recall` -- MQL context recall
- `POST /v1/edges` -- create graph edge
- `GET /v1/ws/stream` -- WebSocket cognition stream

Demo data is shown when the server is unavailable.
