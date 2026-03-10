# Document Intelligence System

Multi-agent PDF analysis system with verifiable citations, powered by Google Gemini.

The system extracts content from PDF documents, builds a searchable knowledge base, and answers questions using a pipeline of AI agents that plan, retrieve, reason, and verify answers against source evidence.

## Architecture

```
User Query --> Planner Agent --> Retrieval Agent --> Reasoning Agent --> Verification Agent --> Answer
```

- **Planner** — Decomposes the query into retrieval steps with multi-query search
- **Retrieval** — Executes vector search, table lookup, math calculations, citation checks
- **Reasoning** — Synthesizes evidence into a cited answer
- **Verification** — Validates every claim against source evidence

## Prerequisites

- Docker and Docker Compose
- Google Gemini API key

## Setup

```bash
# Configure
echo "GEMINI_API_KEY=your_key_here" > .env
echo "GEMINI_MODEL=gemini-2.0-flash" >> .env

# Add PDF
cp your_report.pdf ./data/

# Build (only needed once, or when requirements.txt changes)
docker compose build

# Run ETL (once per PDF)
docker compose --profile etl run etl

# Start API
docker compose up -d
```

## Usage

### Query

```bash
curl -s -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How many cyber security firms are in Ireland?"}' | python3 -m json.tool
```

Response:

```json
{
    "answer": "There are 489 firms offering cyber security products or services in Ireland (Page 4).",
    "citations": [{"page": 4, "text": "There are 489 firms offering cyber security..."}],
    "trace_id": "a1b2c3d4-...",
    "verified": true,
    "confidence": 1.0
}
```

### Health Check

```bash
curl -s http://localhost:8000/health | python3 -m json.tool
```

### View Traces

```bash
# List recent traces
curl -s http://localhost:8000/traces | python3 -m json.tool

# Get specific trace
curl -s http://localhost:8000/traces/{trace_id} | python3 -m json.tool
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | /query | Ask a question about the document |
| GET | /health | System status and store counts |
| GET | /traces | List recent query traces |
| GET | /traces/{id} | Full execution trace for a query |

## Development

Code directories are mounted as Docker volumes. The API server runs with auto-reload, so code changes take effect immediately without rebuilding or restarting.

```bash
# Edit any file -- changes apply automatically
vim agents/reasoning_agent.py

# Only rebuild if requirements.txt changes
docker compose build

# Re-index if PDF changes
rm -rf chroma_store/ data/structured.db
docker compose --profile etl run etl
```

## Project Structure

```
docker/Dockerfile          # Dependencies and base image
docker-compose.yml         # Services with volume mounts
requirements.txt           # Python packages
run_etl.py                 # ETL entry point
etl/                       # PDF extraction, chunking, loading
agents/                    # Planner, retrieval, reasoning, verification
tools/                     # Vector search, table lookup, math, citations
storage/                   # ChromaDB and DuckDB wrappers
api/                       # FastAPI server
data/                      # PDF files (mounted volume)
chroma_store/              # Vector database (mounted volume)
logs/                      # Query traces (mounted volume)
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| GEMINI_API_KEY | required | Google Gemini API key |
| GEMINI_MODEL | gemini-2.0-flash | Model to use |

## Limitations

- Processes one PDF at a time
- Free-tier Gemini rate limits may slow responses
- Page numbers in citations are sequential PDF pages, not printed page numbers
