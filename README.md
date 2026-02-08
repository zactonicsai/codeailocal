# Zactonics AI Spring Forge — Complete Technical Tutorial

> **A line-by-line walkthrough of every file, every data flow, and every design decision in the project.**

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture and Data Flow](#2-architecture-and-data-flow)
3. [File-by-File Deep Dive](#3-file-by-file-deep-dive)
   - 3.1 [docker-compose.yml — Orchestration Layer](#31-docker-composeyml--orchestration-layer)
   - 3.2 [nginx.conf — Reverse Proxy & Routing](#32-nginxconf--reverse-proxy--routing)
   - 3.3 [langchain-service/Dockerfile — RAG Container Build](#33-langchain-servicedockerfile--rag-container-build)
   - 3.4 [langchain-service/requirements.txt — Python Dependencies](#34-langchain-servicerequirementstxt--python-dependencies)
   - 3.5 [langchain-service/app.py — The RAG Engine (Line-by-Line)](#35-langchain-serviceapppy--the-rag-engine-line-by-line)
   - 3.6 [langchain-service/snippets/spring_best_practices.json — The Knowledge Base](#36-langchain-servicesnippetsspring_best_practicesjson--the-knowledge-base)
   - 3.7 [spring-coder.html — The Frontend (Line-by-Line)](#37-spring-coderhtml--the-frontend-line-by-line)
   - 3.8 [setup.sh — CLI Bootstrap Script](#38-setupsh--cli-bootstrap-script)
4. [Data Flow Walkthroughs](#4-data-flow-walkthroughs)
   - 4.1 [Startup Sequence](#41-startup-sequence)
   - 4.2 [Direct Generation (RAG Off)](#42-direct-generation-rag-off)
   - 4.3 [RAG-Enhanced Generation (RAG On)](#43-rag-enhanced-generation-rag-on)
   - 4.4 [Snippet Seeding into ChromaDB](#44-snippet-seeding-into-chromadb)
5. [Component Summary](#5-component-summary)
6. [Dependency Reference](#6-dependency-reference)
7. [Links and Further Reading](#7-links-and-further-reading)

---

## 1. Project Overview

**Zactonics AI Spring Forge** is a fully local, offline, AI-powered Java Spring Boot code generator. It combines four containerized services to provide context-aware code generation that follows your team's best practices.

The project has **two generation modes**:

**Direct Mode (RAG Off):** The browser sends a prompt directly to Ollama's LLM. The model generates code using only its pre-trained knowledge — fast, but generic.

**RAG Mode (RAG On):** The browser sends the prompt to a LangChain FastAPI service. That service embeds the prompt, searches ChromaDB for the most relevant Java Spring best-practice code snippets, injects those snippets into the LLM's system prompt as context, and then streams the enriched output back. The model now generates code that follows your specific patterns and conventions.

**Why RAG matters:** Small local models (0.5B–7B parameters) lack the depth to consistently produce idiomatic Spring Boot code. By injecting real, annotated best-practice examples into every prompt, even a tiny model can produce code that uses `@RequiredArgsConstructor` instead of `@Autowired`, returns `ResponseEntity` instead of raw objects, applies `@Transactional(readOnly = true)` by default, and includes `// Best Practice:` comments explaining *why*.

---

## 2. Architecture and Data Flow

```
                        ┌─────────────────────────────┐
                        │       User's Browser         │
                        │   http://localhost:8080       │
                        └─────────────┬───────────────┘
                                      │
                              ┌───────▼────────┐
                              │  nginx :8080    │
                              │  (spring-forge- │
                              │   ui container) │
                              └───┬────────┬───┘
                 ┌────────────────┘        └─────────────────┐
                 │ /ollama/*                                  │ /rag/*
                 │ (Direct mode)                              │ (RAG mode)
                 ▼                                            ▼
      ┌─────────────────┐                        ┌──────────────────────┐
      │  Ollama :11434   │◄───────────────────────│  LangChain RAG      │
      │                  │   3. Send enriched     │  FastAPI :8100       │
      │  • qwen2.5-coder │      prompt to LLM    │                      │
      │  • nomic-embed-  │                        │  1. Embed user       │
      │    text          │   4. Stream tokens     │     prompt           │
      │                  │──────────────────────▶ │  2. Query ChromaDB   │
      └─────────────────┘                        │  5. Stream SSE back  │
                 ▲                                └───────────┬──────────┘
                 │ Embeddings API                             │
                 │ POST /api/embed                            │ Cosine similarity
                 │                                            │ search
                 │                                ┌───────────▼──────────┐
                 └────────────────────────────────│  ChromaDB :8200      │
                                                  │                      │
                                                  │  Collection:         │
                                                  │  spring_best_        │
                                                  │  practices           │
                                                  │  (10 snippets with   │
                                                  │   768-dim vectors)   │
                                                  └──────────────────────┘
```

**Port Map:**

| Port  | Service              | Internal Port | Purpose                              |
|-------|----------------------|---------------|--------------------------------------|
| 8080  | nginx (spring-forge-ui) | 80         | Web UI + reverse proxy               |
| 8100  | langchain-rag        | 8100          | RAG API (FastAPI)                    |
| 8200  | chromadb             | 8000          | Vector database HTTP API             |
| 11434 | ollama               | 11434         | LLM inference + embeddings           |

---

## 3. File-by-File Deep Dive

### 3.1 `docker-compose.yml` — Orchestration Layer

This file defines all five services and two persistent volumes. Docker Compose starts them in dependency order and manages their internal DNS so containers can talk to each other by name.

```yaml
services:
```

The top-level `services:` key begins the service definitions. Each child key becomes a container.

---

**Lines 2–25: `ollama` — The LLM Inference Server**

```yaml
  ollama:
    image: ollama/ollama:latest       # Official Ollama Docker image from Docker Hub
    container_name: ollama            # Fixed name so other containers can reference it
    ports:
      - "11434:11434"                 # Expose Ollama API to the host machine
    volumes:
      - ollama_data:/root/.ollama     # Persist downloaded models across restarts
    restart: unless-stopped           # Auto-restart on crash, but not if manually stopped
```

**What this does:** Starts an Ollama server that can load and serve any model from the [Ollama library](https://ollama.com/library). The `ollama_data` volume means downloaded models (which can be several GB) survive `docker compose down` / `up` cycles.

The commented-out GPU sections (lines 12–25) show how to enable NVIDIA or AMD GPU acceleration. Without a GPU, Ollama runs on CPU — functional but slower.

**Key design decision:** The port `11434` is Ollama's default. By exposing it, you can also use `curl http://localhost:11434/api/tags` from the host to debug.

---

**Lines 28–46: `model-pull` — One-Shot Model Downloader**

```yaml
  model-pull:
    image: ollama/ollama:latest
    container_name: ollama-model-pull
    depends_on:
      - ollama                        # Wait for Ollama container to start
    environment:
      - OLLAMA_HOST=http://ollama:11434   # Tell the ollama CLI where the server is
    entrypoint: >
      sh -c '
        echo "Waiting for Ollama to be ready..."
        until ollama list > /dev/null 2>&1; do
          sleep 2                     # Poll every 2 seconds until API responds
        done
        echo "Pulling coding model..."
        ollama run qwen2.5-coder:0.5b --keepalive 0 ""    # Pull + warm up, then unload
        echo "Pulling embedding model for RAG..."
        ollama pull nomic-embed-text  # 768-dimension embedding model for ChromaDB
      '
    restart: "no"                     # Run once and exit — not a long-lived service
```

**What this does:** This is an *init container* pattern. It runs once at startup to download two models:

1. **`qwen2.5-coder:0.5b`** — A tiny coding model (500M params). Good for testing on low-RAM machines. The `--keepalive 0` flag loads the model, runs an empty prompt (to verify it works), then unloads it from VRAM/RAM immediately.

2. **`nomic-embed-text`** — A 137M-parameter embedding model that converts text into 768-dimensional vectors. The RAG pipeline uses this to embed both the code snippets (at seed time) and user prompts (at query time) so ChromaDB can find similar snippets via cosine distance.

**Key design decision:** `restart: "no"` means this container runs once and stops. It won't restart on `docker compose restart`. The `until ollama list` loop handles the race condition where `ollama` container is "started" but the HTTP server isn't ready yet.

---

**Lines 48–62: `chromadb` — The Vector Database**

```yaml
  chromadb:
    image: chromadb/chroma:latest     # Official ChromaDB image
    container_name: chromadb
    ports:
      - "8200:8000"                   # Remap internal 8000 to host 8200 (avoid conflicts)
    volumes:
      - chroma_data:/chroma/chroma    # Persist vector data across restarts
    environment:
      - IS_PERSISTENT=TRUE            # Enable disk-backed storage (not in-memory)
      - ANONYMIZED_TELEMETRY=FALSE    # Disable telemetry for offline/private use
    restart: unless-stopped
```

**What this does:** ChromaDB is an open-source vector database. It stores documents alongside their embedding vectors and supports fast approximate nearest-neighbor (ANN) search using HNSW (Hierarchical Navigable Small World) graphs.

In this project, ChromaDB holds a single **collection** called `spring_best_practices` containing 10 Java Spring code snippets. Each snippet is stored as:
- **Document:** The full text (title + description + Java code)
- **Embedding:** A 768-dimensional vector produced by `nomic-embed-text`
- **Metadata:** Category, title, description (for filtering and display)

**Key design decisions:**
- Port `8200` (not `8000`) avoids conflicts with other services on common development machines.
- `IS_PERSISTENT=TRUE` ensures the seeded vectors survive container restarts. Without this, ChromaDB uses ephemeral in-memory storage.
- `ANONYMIZED_TELEMETRY=FALSE` respects the project's goal of being fully offline.

---

**Lines 64–87: `langchain-rag` — The RAG Orchestration Service**

```yaml
  langchain-rag:
    build:
      context: ./langchain-service    # Build from local Dockerfile
      dockerfile: Dockerfile
    container_name: langchain-rag
    ports:
      - "8100:8100"                   # Expose FastAPI on host port 8100
    environment:
      - OLLAMA_BASE_URL=http://ollama:11434    # Docker internal DNS — not localhost
      - CHROMA_HOST=chromadb                    # Docker internal DNS for ChromaDB
      - CHROMA_PORT=8000                        # ChromaDB's internal port (not 8200)
      - EMBEDDING_MODEL=nomic-embed-text        # Which Ollama model to use for embeddings
    depends_on:
      - ollama
      - chromadb
    restart: unless-stopped
```

**What this does:** Builds and runs the Python FastAPI service defined in `langchain-service/`. This is the brain of the RAG pipeline.

**Key design decisions:**
- `build.context` tells Docker to look in the `langchain-service/` directory for the Dockerfile and application code.
- Environment variables use **Docker's internal DNS names** (`ollama`, `chromadb`) not `localhost`. Inside a Docker network, containers reference each other by their service name.
- `CHROMA_PORT=8000` is the *internal* port, not the host-mapped `8200`. Container-to-container communication uses internal ports.
- `depends_on` ensures Ollama and ChromaDB containers *start* first (but doesn't guarantee they're *ready* — that's why `app.py` has retry loops).

---

**Lines 89–101: `spring-forge-ui` — The Web Frontend**

```yaml
  spring-forge-ui:
    image: nginx:alpine               # Lightweight Alpine-based nginx
    container_name: spring-forge-ui
    ports:
      - "8080:80"                     # Serve on host port 8080
    volumes:
      - ./spring-coder.html:/usr/share/nginx/html/index.html:ro   # Mount HTML as index
      - ./nginx.conf:/etc/nginx/conf.d/default.conf:ro            # Custom nginx config
    depends_on:
      - ollama
      - langchain-rag
    restart: unless-stopped
```

**What this does:** Serves the single-page application (`spring-coder.html`) and acts as a reverse proxy, routing API requests to the appropriate backend service.

**Key design decisions:**
- `:ro` (read-only) mounts prevent the container from modifying your source files.
- `nginx:alpine` is only ~5MB, making startup nearly instant.
- The `depends_on` list ensures both Ollama and the RAG service are at least started before nginx begins accepting traffic.

---

**Lines 103–105: `volumes` — Persistent Storage**

```yaml
volumes:
  ollama_data:    # Stores downloaded LLM models (~500MB to ~30GB per model)
  chroma_data:    # Stores ChromaDB vector data and HNSW indexes (~1MB for 10 snippets)
```

Named volumes persist data independently of container lifecycle. `docker compose down` keeps volumes; `docker compose down -v` destroys them.

---

### 3.2 `nginx.conf` — Reverse Proxy & Routing

```nginx
server {
    listen 80;                          # Listen on container port 80 (mapped to host 8080)
    server_name localhost;
```

**Line 1–3:** Standard nginx server block. Listens on port 80 inside the container.

```nginx
    # Serve the Spring Forge UI
    location / {
        root /usr/share/nginx/html;     # Serve static files from this directory
        index index.html;               # Default file is our mounted spring-coder.html
    }
```

**Lines 5–9: Static file serving.** Any request to `/` serves the Spring Forge HTML file. Since we mounted `spring-coder.html` as `index.html`, this is all that's needed.

```nginx
    # Reverse proxy to Ollama (avoids CORS issues)
    location /ollama/ {
        proxy_pass http://ollama:11434/;   # Forward to Ollama's internal address
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_http_version 1.1;            # Required for streaming
        proxy_set_header Connection '';     # Disable keep-alive Connection header
        proxy_buffering off;               # CRITICAL: Don't buffer — stream through immediately
        proxy_cache off;                   # Don't cache LLM responses
        chunked_transfer_encoding on;      # Support chunked streaming
        proxy_read_timeout 600s;           # 10-minute timeout for long generations
        proxy_send_timeout 600s;
    }
```

**Lines 11–24: Ollama proxy.** This is the **Direct Mode** path. When the frontend (running in the browser at `localhost:8080`) calls `/ollama/api/chat`, nginx strips the `/ollama/` prefix and forwards to `http://ollama:11434/api/chat`.

**Why is this needed?** Browsers enforce CORS (Cross-Origin Resource Security). Without this proxy, the browser at `localhost:8080` would refuse to call `localhost:11434` because they're different origins. The nginx proxy makes it appear as if the Ollama API lives on the same origin.

**Critical streaming settings:**
- `proxy_buffering off` — Without this, nginx would buffer the entire LLM response before sending it to the browser, defeating streaming.
- `proxy_http_version 1.1` — HTTP/1.1 supports chunked transfer encoding needed for streaming.
- `proxy_read_timeout 600s` — Large models can take minutes to generate; the default 60s would cause premature disconnects.

```nginx
    # Reverse proxy to LangChain RAG service
    location /rag/ {
        proxy_pass http://langchain-rag:8100/;   # Forward to the FastAPI service
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_http_version 1.1;
        proxy_set_header Connection '';
        proxy_buffering off;
        proxy_cache off;
        chunked_transfer_encoding on;
        proxy_read_timeout 600s;
        proxy_send_timeout 600s;
    }
```

**Lines 26–38: RAG proxy.** This is the **RAG Mode** path. When the frontend calls `/rag/generate`, nginx forwards to `http://langchain-rag:8100/generate`. Identical streaming settings apply because the RAG service also streams SSE (Server-Sent Events) back to the browser.

---

### 3.3 `langchain-service/Dockerfile` — RAG Container Build

```dockerfile
FROM python:3.12-slim              # Minimal Python 3.12 base image (~150MB)

WORKDIR /app                       # Set working directory inside container

COPY requirements.txt .            # Copy dependency list first (Docker layer caching)
RUN pip install --no-cache-dir -r requirements.txt   # Install all Python packages

COPY app.py .                      # Copy the FastAPI application
COPY snippets/ ./snippets/         # Copy the best-practice snippet JSON

EXPOSE 8100                        # Document that this container listens on 8100

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8100", "--log-level", "info"]
```

**Line-by-line design decisions:**

- **`python:3.12-slim`** — The `slim` variant omits GCC, make, etc. saving ~400MB over the full image. Since all our dependencies are pure Python wheels, no compilation is needed.
- **`COPY requirements.txt` before `COPY app.py`** — This is a Docker layer caching optimization. If you change `app.py` but not `requirements.txt`, Docker reuses the cached `pip install` layer, making rebuilds take seconds instead of minutes.
- **`--no-cache-dir`** — Tells pip not to store downloaded wheels in a cache directory, reducing image size.
- **`uvicorn`** — An ASGI server that runs the FastAPI app. `--host 0.0.0.0` binds to all interfaces (required inside Docker so the host can reach it).

---

### 3.4 `langchain-service/requirements.txt` — Python Dependencies

```
fastapi==0.115.6                    # Modern async web framework for the REST API
uvicorn[standard]==0.34.0           # ASGI server with HTTP/1.1 and WebSocket support
langchain==0.3.14                   # LangChain core — prompt templates, chains, abstractions
langchain-community==0.3.14         # Community integrations (Ollama, ChromaDB connectors)
langchain-chroma==0.2.2             # ChromaDB-specific LangChain integration
langchain-ollama==0.3.0             # Ollama-specific LangChain integration
chromadb-client==0.5.23             # ChromaDB Python HTTP client (not the full server)
httpx==0.28.1                       # Async HTTP client (replaces requests for async support)
pydantic==2.10.4                    # Data validation via type hints (FastAPI's foundation)
```

**Why these specific packages:**

- **`chromadb-client`** (not `chromadb`) — We only need the HTTP client since the server runs in its own container. The full `chromadb` package bundles SQLite, ONNX runtime, and other heavy dependencies we don't need.
- **`httpx`** — Used instead of `requests` because it supports async/await and streaming, which are essential for our SSE (Server-Sent Events) pipeline.
- **`langchain-ollama`** — Provides `OllamaEmbeddings` and `ChatOllama` classes that handle the Ollama-specific API format.
- **`pydantic`** — FastAPI uses Pydantic models for request/response validation. Our `GenerateRequest`, `HealthResponse`, etc. are all Pydantic models.

---

### 3.5 `langchain-service/app.py` — The RAG Engine (Line-by-Line)

This is the heart of the project. Let's walk through every section.

#### Lines 1–10: Module Docstring

```python
"""
LangChain RAG Service for Zactonics AI Spring Forge
====================================================
Provides context-enriched code generation by retrieving relevant Java Spring
best-practice snippets from ChromaDB and injecting them into the LLM prompt.

Architecture:
  [User Prompt] → [FastAPI] → [ChromaDB similarity search] → [Build context]
       → [LangChain + Ollama] → [Streamed Java code response]
"""
```

This documents the RAG (Retrieval-Augmented Generation) pattern. The key insight: instead of relying solely on the LLM's pre-trained knowledge, we *retrieve* relevant examples first and *augment* the prompt with them before *generating*.

---

#### Lines 12–25: Imports

```python
import json                          # Parse JSON snippets and SSE events
import logging                       # Structured logging for debugging
import os                            # Read environment variables
import time                          # Sleep in retry loops
from contextlib import asynccontextmanager  # FastAPI lifespan manager
from pathlib import Path             # File path handling
from typing import Optional          # Type hints

import chromadb                      # ChromaDB Python client
import httpx                         # Async HTTP client for Ollama API calls
from fastapi import FastAPI, HTTPException          # Web framework
from fastapi.middleware.cors import CORSMiddleware   # Cross-origin support
from fastapi.responses import StreamingResponse      # SSE streaming
from pydantic import BaseModel, Field               # Request/response validation
```

---

#### Lines 27–37: Configuration Constants

```python
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
CHROMA_HOST = os.getenv("CHROMA_HOST", "chromadb")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
COLLECTION_NAME = "spring_best_practices"
SNIPPETS_PATH = Path("/app/snippets/spring_best_practices.json")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("langchain-rag")
```

**Every value is configurable via environment variables** with sensible defaults. This follows the [12-Factor App](https://12factor.net/config) methodology — the same code runs in Docker, on bare metal, or in CI by changing environment variables.

- `OLLAMA_BASE_URL` defaults to the Docker internal DNS name `ollama`.
- `COLLECTION_NAME` is the ChromaDB collection where snippets are stored.
- `SNIPPETS_PATH` points to where the Dockerfile copies the JSON file.

---

#### Lines 39–42: Global State

```python
chroma_client: Optional[chromadb.HttpClient] = None
collection: Optional[chromadb.Collection] = None
```

These module-level variables are initialized during the FastAPI lifespan and shared across all request handlers. This avoids creating a new ChromaDB connection per request.

---

#### Lines 47–58: `wait_for_chroma()` — Retry Loop

```python
def wait_for_chroma(max_retries: int = 30, delay: float = 2.0):
    """Block until ChromaDB is reachable."""
    for attempt in range(max_retries):
        try:
            client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
            client.heartbeat()                  # Simple health check
            logger.info("✓ ChromaDB is ready (attempt %d)", attempt + 1)
            return client
        except Exception:
            logger.info("Waiting for ChromaDB... (attempt %d/%d)", attempt + 1, max_retries)
            time.sleep(delay)
    raise RuntimeError("ChromaDB not reachable after retries")
```

**Why retry loops?** Docker Compose `depends_on` only waits for the container to *start*, not for the service inside it to be *ready*. ChromaDB might take 3–10 seconds to initialize its HNSW index and start accepting HTTP connections. This function polls every 2 seconds for up to 60 seconds.

---

#### Lines 61–74: `wait_for_ollama()` — Retry Loop

```python
def wait_for_ollama(max_retries: int = 30, delay: float = 3.0):
    """Block until Ollama is reachable, then pull the embedding model."""
    import httpx as hx
    for attempt in range(max_retries):
        try:
            r = hx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
            if r.status_code == 200:
                logger.info("✓ Ollama is ready (attempt %d)", attempt + 1)
                return
        except Exception:
            pass
        logger.info("Waiting for Ollama... (attempt %d/%d)", attempt + 1, max_retries)
        time.sleep(delay)
    raise RuntimeError("Ollama not reachable after retries")
```

Same pattern as ChromaDB. Ollama's `/api/tags` endpoint lists available models and is the lightest health check available.

---

#### Lines 77–87: `get_embedding()` — The Vector Generator

```python
def get_embedding(text: str) -> list[float]:
    """Get embedding vector from Ollama's embedding endpoint."""
    resp = httpx.post(
        f"{OLLAMA_BASE_URL}/api/embed",
        json={"model": EMBEDDING_MODEL, "input": text},
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["embeddings"][0]
```

**This is a critical function.** It converts text into a 768-dimensional float vector using the `nomic-embed-text` model running in Ollama. This vector represents the *semantic meaning* of the text. Texts with similar meanings have vectors that are close in cosine distance.

**Data flow:**
1. Input: A string like `"REST Controller Best Practice\nProduction-grade REST controller..."`
2. Sends HTTP POST to `http://ollama:11434/api/embed`
3. Ollama runs the text through the `nomic-embed-text` model
4. Returns: `{"embeddings": [[0.0234, -0.0891, 0.1456, ...]]}` — 768 floats
5. Function returns the inner list: `[0.0234, -0.0891, 0.1456, ...]`

---

#### Lines 90–122: `seed_snippets()` — Loading Knowledge into ChromaDB

```python
def seed_snippets(coll: chromadb.Collection):
    """Load Java Spring best-practice snippets into ChromaDB if not already present."""
    existing = coll.count()
    if existing > 0:                          # Idempotent: skip if already seeded
        logger.info("Collection already has %d documents, skipping seed.", existing)
        return

    if not SNIPPETS_PATH.exists():
        logger.warning("Snippets file not found at %s", SNIPPETS_PATH)
        return

    snippets = json.loads(SNIPPETS_PATH.read_text())   # Load the JSON file
    logger.info("Seeding %d snippets into ChromaDB...", len(snippets))

    ids = []
    documents = []
    metadatas = []
    embeddings = []

    for s in snippets:
        doc_text = f"{s['title']}\n{s['description']}\n\n{s['code']}"   # Combine fields
        ids.append(s["id"])
        documents.append(doc_text)
        metadatas.append({
            "category": s["category"],
            "title": s["title"],
            "description": s["description"],
        })
        emb = get_embedding(doc_text[:2000])     # Truncate to embedding model's context limit
        embeddings.append(emb)

    coll.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
    logger.info("✓ Seeded %d snippets into collection '%s'", len(snippets), COLLECTION_NAME)
```

**Step-by-step data flow:**

1. **Check:** If the collection already has documents, skip (idempotent).
2. **Load:** Read the 10 snippets from `spring_best_practices.json`.
3. **For each snippet:**
   a. Concatenate title + description + code into one string.
   b. Send the first 2000 characters to Ollama's `/api/embed` endpoint.
   c. Receive a 768-dimensional vector back.
4. **Batch insert:** `coll.add()` sends all 10 documents, vectors, and metadata to ChromaDB in one call. ChromaDB builds an HNSW index over the vectors for fast search.

**Why truncate at 2000 chars?** The `nomic-embed-text` model has a context window of ~8192 tokens, but embedding quality degrades for very long documents. 2000 characters captures the title, description, and key patterns of each snippet while staying well within limits.

---

#### Lines 127–158: `lifespan()` — Application Startup/Shutdown

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: connect to ChromaDB + Ollama, seed data. Shutdown: cleanup."""
    global chroma_client, collection

    logger.info("Starting LangChain RAG service...")
    wait_for_ollama()                            # Block until Ollama is ready

    # Pull embedding model if not present
    logger.info("Ensuring embedding model '%s' is available...", EMBEDDING_MODEL)
    try:
        httpx.post(
            f"{OLLAMA_BASE_URL}/api/pull",
            json={"name": EMBEDDING_MODEL, "stream": False},
            timeout=600,                         # May take 5+ minutes to download
        )
        logger.info("✓ Embedding model ready")
    except Exception as e:
        logger.warning("Could not pull embedding model: %s", e)

    chroma_client = wait_for_chroma()            # Block until ChromaDB is ready
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},       # Use cosine distance metric
    )

    seed_snippets(collection)                    # Seed snippets if empty
    logger.info("✓ LangChain RAG service ready")

    yield                                        # Application runs here

    logger.info("Shutting down LangChain RAG service")
```

**This is FastAPI's lifespan protocol.** Everything before `yield` runs on startup; everything after `yield` runs on shutdown.

**Startup sequence:**
1. Wait for Ollama → pull embedding model (if not cached) → wait for ChromaDB → create/get collection → seed snippets.
2. `hnsw:space: "cosine"` tells ChromaDB to use cosine similarity (not Euclidean distance) for nearest-neighbor search. Cosine similarity is standard for text embeddings because it measures angle between vectors, ignoring magnitude.

---

#### Lines 163–175: FastAPI App Definition

```python
app = FastAPI(
    title="Spring Forge RAG API",
    description="RAG-powered Java Spring code generation with best-practice context",
    version="1.0.0",
    lifespan=lifespan,                 # Attach our startup/shutdown hooks
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],               # Allow all origins (OK for local-only service)
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**CORS middleware** is configured permissively because this is a local service. In production, you'd whitelist specific origins.

---

#### Lines 180–205: Pydantic Request/Response Models

```python
class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="User's code generation prompt")
    model: str = Field(default="qwen2.5-coder:0.5b", description="Ollama model name")
    top_k: int = Field(default=3, ge=1, le=10, description="Number of snippets to retrieve")
    stream: bool = Field(default=True, description="Stream the response")
    category: Optional[str] = Field(default=None, description="Filter snippets by category")
```

- **`prompt`** — Required (the `...` means no default). This is the user's natural-language request.
- **`model`** — Which Ollama model to use for generation. Defaults to the small 0.5B model.
- **`top_k`** — How many snippets to retrieve from ChromaDB. Default 3 means the top 3 most relevant snippets are injected into the prompt. Higher values provide more context but increase prompt length.
- **`category`** — Optional filter. If set to `"rest"`, ChromaDB only searches within REST controller snippets.

---

#### Lines 210–235: `/health` Endpoint

```python
@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check for the RAG service."""
```

This endpoint checks connectivity to both ChromaDB and Ollama and reports the number of loaded snippets. The frontend calls this when the user toggles RAG on to show "10 snippets" in the status badge.

---

#### Lines 238–266: `/search` Endpoint

```python
@app.post("/search", response_model=SearchResponse)
async def search_snippets(prompt: str, top_k: int = 5, category: Optional[str] = None):
```

A utility endpoint for searching snippets *without* generating code. Useful for debugging which snippets would be retrieved for a given prompt.

**Data flow:**
1. Embed the query: `get_embedding(prompt)` → 768-dim vector
2. Query ChromaDB: `collection.query(query_embeddings=[...], n_results=top_k)`
3. ChromaDB runs HNSW approximate nearest-neighbor search
4. Return results with cosine similarity scores: `score = 1 - distance`

---

#### Lines 269–333: `/generate` Endpoint — The Main RAG Pipeline

This is the most important function. Let's trace the complete data flow:

**Step 1: Retrieve relevant snippets (lines 280–289)**

```python
    query_embedding = get_embedding(req.prompt)     # Embed user's prompt
    where_filter = {"category": req.category} if req.category else None

    results = collection.query(
        query_embeddings=[query_embedding],          # The prompt's vector
        n_results=req.top_k,                         # Get top 3 matches
        where=where_filter,                          # Optional category filter
        include=["documents", "metadatas"],           # Return full text + metadata
    )
```

**What happens inside ChromaDB:** The HNSW index finds the `top_k` documents whose embedding vectors are closest (by cosine distance) to the prompt's embedding vector. If the user asks about "REST controller", the vectors for the REST controller snippet and the DTO snippet will be closest.

**Step 2: Build context block (lines 291–302)**

```python
    context_blocks = []
    for i, doc in enumerate(results["documents"][0]):
        meta = results["metadatas"][0][i]
        context_blocks.append(
            f"=== Best Practice Reference: {meta.get('title', 'Snippet')} ===\n"
            f"Category: {meta.get('category', 'general')}\n"
            f"{meta.get('description', '')}\n\n"
            f"{doc}\n"
        )
    context = "\n".join(context_blocks)
```

This assembles the retrieved snippets into a single text block. For a REST controller query, the context might include the REST Controller snippet, the Exception Handler snippet, and the DTO snippet — the three most semantically similar.

**Step 3: Build enriched system prompt (lines 305–324)**

```python
    system_prompt = f"""You are a senior Java Spring Boot developer...

REFERENCE CONTEXT — Use these best-practice examples as guidance...

{context}                      # <── The retrieved snippets are injected HERE

RULES:
- Output ONLY the Java source code...
- Include "// Best Practice:" comments throughout the code...
"""
```

**This is the core RAG innovation.** The system prompt now contains 3 real Java code examples that demonstrate the exact patterns, annotations, and commenting style we want. Even a small 0.5B model can follow these examples because they're right there in the prompt — no pre-training required.

**Step 4: Stream to Ollama (lines 326–333)**

```python
    if req.stream:
        return StreamingResponse(
            _stream_ollama(req.model, system_prompt, req.prompt),
            media_type="text/event-stream",
        )
```

The response is returned as a `StreamingResponse` with `text/event-stream` MIME type (Server-Sent Events protocol).

---

#### Lines 336–366: `_stream_ollama()` — SSE Streaming Bridge

```python
async def _stream_ollama(model: str, system_prompt: str, user_prompt: str):
    """Stream tokens from Ollama's chat API as SSE events."""
    async with httpx.AsyncClient(timeout=httpx.Timeout(300)) as client:
        async with client.stream(
            "POST",
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": model,
                "stream": True,
                "messages": [
                    {"role": "system", "content": system_prompt},   # Enriched prompt
                    {"role": "user", "content": user_prompt},       # Original question
                ],
            },
        ) as response:
            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    if data.get("message", {}).get("content"):
                        sse_event = {
                            "type": "content_block_delta",
                            "delta": {"text": data["message"]["content"]},
                        }
                        yield f"data: {json.dumps(sse_event)}\n\n"
                    if data.get("done"):
                        yield "data: [DONE]\n\n"
                except json.JSONDecodeError:
                    continue
```

**This function bridges two streaming protocols:**

1. **Ollama's format:** Each line is a JSON object: `{"message": {"content": "pub"}, "done": false}`
2. **SSE format expected by the frontend:** Each event is `data: {"type": "content_block_delta", "delta": {"text": "pub"}}\n\n`

The function reads tokens one-by-one from Ollama, repackages them into SSE events matching the Anthropic Messages API format (so the frontend's existing parser works with zero changes), and yields them.

**Why async generators?** FastAPI's `StreamingResponse` accepts an async generator. Each `yield` sends one SSE event to the browser immediately. The browser receives tokens as they're generated, creating the typewriter effect.

---

#### Lines 392–425: Snippet Management Endpoints

```python
@app.get("/snippets")                    # List all loaded snippets
@app.post("/snippets/reload")            # Force reload from disk (hot-reload)
```

The `/snippets/reload` endpoint deletes the ChromaDB collection, recreates it, and re-seeds from the JSON file. This lets you edit snippets and refresh without restarting any containers.

---

### 3.6 `langchain-service/snippets/spring_best_practices.json` — The Knowledge Base

This JSON file contains 10 Java Spring best-practice code snippets. Each snippet follows this schema:

```json
{
  "id": "rest-controller-best-practice",       // Unique ID for ChromaDB
  "category": "rest",                           // Filter category
  "title": "REST Controller Best Practice",     // Human-readable title
  "description": "Production-grade REST...",    // Short description
  "code": "package com.example...              // Full Java source code
           // Best Practice: Use @RequiredArgsConstructor for constructor injection...
           "
}
```

**The 10 included snippets:**

| # | ID | Category | What It Teaches |
|---|-----|----------|-----------------|
| 1 | `rest-controller-best-practice` | `rest` | `ResponseEntity`, `@Valid`, `@PageableDefault`, thin controllers, OpenAPI annotations |
| 2 | `jpa-entity-best-practice` | `entity` | UUID PKs, `@EntityListeners`, `BigDecimal` for money, `@Version` optimistic locking, proper `equals/hashCode` |
| 3 | `service-layer-best-practice` | `service` | `@Transactional(readOnly=true)` default, domain exceptions, DTO mapping at service boundary, SLF4J logging levels |
| 4 | `repository-best-practice` | `repo` | Derived queries, `@Query` JPQL, projections, `@Modifying` bulk updates, `JpaSpecificationExecutor` |
| 5 | `dto-mapper-best-practice` | `dto` | Separate Create/Update/Response DTOs, `@Valid` nesting, MapStruct `@Mapper(componentModel="spring")`, `@MappingTarget` |
| 6 | `security-config-best-practice` | `security` | `SecurityFilterChain` bean, stateless JWT, BCrypt(12), CORS whitelist, `@EnableMethodSecurity` |
| 7 | `unit-test-best-practice` | `test` | `@ExtendWith(MockitoExtension)`, BDD given/when/then, `@Nested`, AssertJ, `verify(never())` |
| 8 | `exception-handler-best-practice` | `exception` | `@RestControllerAdvice`, structured `ErrorResponse`, field-level validation errors, catch-all handler |
| 9 | `application-config-best-practice` | `config` | `application.yml`, HikariCP tuning, `open-in-view: false`, actuator, `ddl-auto: validate`, batch fetching |
| 10 | `integration-test-best-practice` | `test` | `@Testcontainers`, `@ServiceConnection`, `TestRestTemplate`, `@ActiveProfiles("test")` |

**Every snippet contains `// Best Practice:` inline comments** that explain *why* a particular pattern is used. When these snippets are injected into the LLM prompt, the model learns to generate code with similar comments.

---

### 3.7 `spring-coder.html` — The Frontend (Line-by-Line)

This is a single-file application: HTML, CSS, and JavaScript in one file. Let's cover the key sections.

#### Lines 1–45: HTML Head — Tailwind + Fonts + Theme Configuration

```html
<script src="https://cdn.tailwindcss.com"></script>
```

Loads Tailwind CSS from CDN for utility-class styling. The `tailwind.config` block (lines 11–44) extends the default theme with custom color palettes:
- **`spring`** — Green shades based on Spring's brand color `#6db33f`
- **`forge`** — Dark neutral grays for the dark theme

#### Lines 46–110: Custom CSS

Key styles:
- **`cursor-blink`** — Adds a blinking `▌` cursor during streaming
- **`grain::before`** — Adds a subtle film-grain texture overlay
- **`glow-ring`** — Green glow on the input area when focused
- **`status-pulse`** — Pulsing animation on the connection status dot

#### Lines 131–156: Header Controls

```html
<!-- Connection status -->
<div id="status-badge" ... onclick="checkConnection()">

<!-- RAG toggle -->
<button id="rag-toggle" onclick="toggleRag()">

<!-- RAG status badge (hidden when RAG off) -->
<div id="rag-status-badge" class="hidden">

<!-- Model selector (dynamically populated from Ollama) -->
<select id="model-select">
```

The header contains four interactive elements. The model selector starts with hardcoded options but gets dynamically replaced by `checkConnection()` with whatever models Ollama actually has installed.

#### Lines 163–177: Template Chips

```html
<button onclick="useTemplate('rest')" class="chip">REST Controller</button>
<button onclick="useTemplate('entity')" class="chip">JPA Entity</button>
...
<button onclick="useTemplate('events')" class="chip">Event Driven</button>
<button onclick="useTemplate('config')" class="chip">App Config</button>
```

10 clickable chips that populate the textarea with pre-built prompts. These map to the `templates` JavaScript object.

#### Lines 352–367: JavaScript State Initialization

```javascript
const defaultUrl = window.location.port === '8080'
  ? `${window.location.origin}/ollama`       // Docker: use nginx proxy
  : 'http://localhost:11434';                  // Local dev: direct Ollama
const ragBaseUrl = window.location.port === '8080'
  ? `${window.location.origin}/rag`           // Docker: use nginx proxy
  : 'http://localhost:8100';                   // Local dev: direct RAG service
```

**Auto-detection logic:** If the page is served from port 8080 (Docker nginx), API calls go through the nginx proxy (`/ollama/*`, `/rag/*`). If opened directly as a file or from a dev server on a different port, calls go directly to `localhost`.

```javascript
let ragEnabled = localStorage.getItem('rag-enabled') === 'true';
```

RAG state persists across browser sessions via `localStorage`.

#### Lines 372–446: Template Definitions

```javascript
const templates = {
  rest: `Create a Spring Boot REST controller for a Product entity with:
- CRUD endpoints (GET all with pagination, GET by id, POST, PUT, DELETE)
- Request validation using @Valid
- Proper HTTP status codes and ResponseEntity
- Swagger/OpenAPI annotations`,
  // ... 9 more templates
};
```

Each template is a multi-line prompt designed to trigger the corresponding snippet category in ChromaDB when RAG is enabled. For example, the `rest` template mentions "controller", "endpoints", and "ResponseEntity" — words that will be semantically close to the REST controller snippet's embedding.

#### Lines 461–498: RAG Toggle Functions

```javascript
function toggleRag() {
  ragEnabled = !ragEnabled;
  localStorage.setItem('rag-enabled', ragEnabled);
  updateRagUI();
  if (ragEnabled) checkRagHealth();
}
```

When the user clicks the RAG button:
1. Flip the boolean
2. Persist to localStorage
3. Update button styling (gray → green)
4. If enabling, call `/rag/health` to show snippet count

```javascript
async function checkRagHealth() {
  const res = await fetch(`${ragBaseUrl}/health`, { signal: AbortSignal.timeout(5000) });
  const data = await res.json();
  document.getElementById('rag-snippet-count').textContent = `${data.snippet_count} snippets`;
}
```

Shows "10 snippets" next to the RAG toggle, confirming the service is connected and has data.

#### Lines 544–598: `sendPrompt()` — The Router

```javascript
async function sendPrompt() {
  // ... setup UI (spinner, cursor, clear output)

  try {
    if (ragEnabled) {
      await ragGenerate(prompt, model, codeEl);     // RAG path
    } else {
      await directGenerate(prompt, model, codeEl);  // Direct path
    }
  } catch (e) { ... }

  // ... cleanup UI (restore button, show line count)
}
```

This function routes to one of two code paths based on the `ragEnabled` flag. Both paths ultimately stream SSE events back to `renderCode()`.

#### Lines 600–648: `ragGenerate()` — RAG Mode

```javascript
async function ragGenerate(prompt, model, codeEl) {
  const category = detectCategory(prompt);       // Auto-detect snippet category

  const res = await fetch(`${ragBaseUrl}/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    signal: abortController.signal,              // Supports cancellation via Escape
    body: JSON.stringify({
      prompt: prompt,
      model: model,
      stream: true,
      top_k: 3,
      category: category,                        // Narrows the ChromaDB search
    })
  });
```

**Data flow:**
1. `detectCategory("Create a REST controller...")` returns `"rest"`
2. POST to `/rag/generate` → nginx proxies to `langchain-rag:8100/generate`
3. FastAPI embeds the prompt, searches ChromaDB, builds enriched system prompt, streams from Ollama
4. SSE events flow back through: Ollama → FastAPI → nginx → browser

The SSE parsing loop (lines 623–648) reads each `data: {...}` line, parses the JSON, extracts `event.delta.text`, appends to `generatedCode`, and calls `renderCode()` to update the display.

#### Lines 715–728: `detectCategory()` — Keyword-Based Category Routing

```javascript
function detectCategory(prompt) {
  const lower = prompt.toLowerCase();
  if (lower.includes('controller') || lower.includes('rest') || lower.includes('endpoint')) return 'rest';
  if (lower.includes('entity') || lower.includes('jpa') || lower.includes('table')) return 'entity';
  // ... more categories
  return null;     // No filter — search all snippets
}
```

This is a simple heuristic that maps keywords in the user's prompt to snippet categories. When a category is detected, ChromaDB's `where` filter narrows the search to only that category's snippets, improving relevance.

---

### 3.8 `setup.sh` — CLI Bootstrap Script

This script is for users who want to use **Claude Code** (Anthropic's terminal tool) with local Ollama. It's separate from the Docker-based UI workflow.

```bash
#!/usr/bin/env bash
set -euo pipefail    # Exit on error, undefined vars, pipe failures
```

The script:
1. Starts Ollama via Docker Compose
2. Pulls a coding model (default: `qwen3-coder`)
3. Pre-warms the model (loads it into memory to avoid cold-start timeouts)
4. Checks for Claude Code installation
5. Prints environment variable instructions for connecting Claude Code to local Ollama

---

## 4. Data Flow Walkthroughs

### 4.1 Startup Sequence

```
Time 0s:  docker compose up -d
          ├── ollama container starts (downloads Ollama server image)
          ├── chromadb container starts (initializes HNSW index)
          ├── model-pull container starts (waits for Ollama)
          ├── langchain-rag container starts (waits for Ollama + ChromaDB)
          └── spring-forge-ui container starts (nginx ready immediately)

Time ~5s: Ollama HTTP server is ready on :11434
          ├── model-pull: begins downloading qwen2.5-coder:0.5b (~350MB)
          └── langchain-rag: detects Ollama is ready, pulls nomic-embed-text (~275MB)

Time ~10s: ChromaDB HTTP server is ready on :8000
           └── langchain-rag: detects ChromaDB is ready
                ├── Creates collection "spring_best_practices" (cosine metric)
                └── Begins seeding 10 snippets

Time ~30s: Embedding generation completes
           └── langchain-rag: all 10 snippets embedded and stored in ChromaDB
                └── Logs "✓ LangChain RAG service ready"

Time ~60s: model-pull: finishes downloading qwen2.5-coder:0.5b
           └── Pulls nomic-embed-text (already cached by langchain-rag)
           └── Container exits (restart: "no")

Time ~60s: User opens http://localhost:8080
           └── Browser loads spring-coder.html from nginx
                ├── checkConnection() → GET /ollama/api/tags → "Online (2 models)"
                └── If RAG enabled: checkRagHealth() → GET /rag/health → "10 snippets"
```

### 4.2 Direct Generation (RAG Off)

```
User types: "Create a REST controller for Products"
User clicks: [Generate]

Browser (spring-coder.html)
  │
  ├─ sendPrompt() called
  ├─ ragEnabled === false → calls directGenerate()
  ├─ Builds system prompt (static, no context injection)
  │
  ├─ POST /ollama/v1/messages  ─────────────────────────────────────────┐
  │   body: {model: "qwen2.5-coder:0.5b", stream: true,                │
  │          system: "You are a senior Java Spring Boot developer...",   │
  │          messages: [{role: "user", content: "Create a REST..."}]}   │
  │                                                                      │
  │   ┌─────────── nginx ───────────┐                                    │
  │   │ strips /ollama/ prefix      │                                    │
  │   │ proxy_pass → ollama:11434   │                                    │
  │   └─────────────────────────────┘                                    │
  │                                                                      │
  │   ┌─────────── Ollama ──────────┐                                    │
  │   │ Loads model into memory     │                                    │
  │   │ Generates tokens one by one │                                    │
  │   │ Streams SSE events:         │                                    │
  │   │   data: {"type":"content_block_delta","delta":{"text":"pack"}}   │
  │   │   data: {"type":"content_block_delta","delta":{"text":"age "}}   │
  │   │   ...                       │                                    │
  │   │   data: [DONE]              │                                    │
  │   └─────────────────────────────┘                                    │
  │                                                                      │
  ├─ For each SSE event:                                                 │
  │   ├─ Parse JSON                                                      │
  │   ├─ Append delta.text to generatedCode                              │
  │   └─ renderCode(generatedCode) → updates <code> element + line nums  │
  │                                                                      │
  └─ On [DONE]: remove cursor-blink, show "Done — 85 lines"
```

### 4.3 RAG-Enhanced Generation (RAG On)

```
User types: "Create a REST controller for Products"
User clicks: [Generate]

Browser (spring-coder.html)
  │
  ├─ sendPrompt() called
  ├─ ragEnabled === true → calls ragGenerate()
  ├─ detectCategory("Create a REST controller...") → "rest"
  │
  ├─ POST /rag/generate  ───────────────────────────────────────────────┐
  │   body: {prompt: "Create a REST controller...",                      │
  │          model: "qwen2.5-coder:0.5b", stream: true,                 │
  │          top_k: 3, category: "rest"}                                 │
  │                                                                      │
  │   ┌─────────── nginx ───────────┐                                    │
  │   │ strips /rag/ prefix         │                                    │
  │   │ proxy_pass → langchain-     │                                    │
  │   │   rag:8100                  │                                    │
  │   └─────────────────────────────┘                                    │
  │                                                                      │
  │   ┌─────────── FastAPI (app.py) ── generate_code() ──────────────┐  │
  │   │                                                                │  │
  │   │  STEP 1: Embed the prompt                                      │  │
  │   │  ├─ POST http://ollama:11434/api/embed                         │  │
  │   │  │   body: {model: "nomic-embed-text",                         │  │
  │   │  │          input: "Create a REST controller for Products"}    │  │
  │   │  └─ Returns: [0.0234, -0.0891, 0.1456, ...] (768 floats)     │  │
  │   │                                                                │  │
  │   │  STEP 2: Search ChromaDB                                       │  │
  │   │  ├─ collection.query(                                          │  │
  │   │  │     query_embeddings=[[0.0234, -0.0891, ...]],             │  │
  │   │  │     n_results=3,                                            │  │
  │   │  │     where={"category": "rest"})                             │  │
  │   │  │                                                             │  │
  │   │  │  ChromaDB HNSW search finds nearest vectors:                │  │
  │   │  │  ┌──────────────────────────────────────────────┐           │  │
  │   │  │  │ #1: rest-controller-best-practice  (0.92)    │           │  │
  │   │  │  │ #2: dto-mapper-best-practice       (0.78)    │           │  │
  │   │  │  │ #3: exception-handler-best-practice (0.71)   │           │  │
  │   │  │  └──────────────────────────────────────────────┘           │  │
  │   │  └─ Returns full code text + metadata for top 3                │  │
  │   │                                                                │  │
  │   │  STEP 3: Build enriched system prompt                          │  │
  │   │  ├─ system_prompt = """                                        │  │
  │   │  │    You are a senior Java Spring Boot developer...           │  │
  │   │  │                                                             │  │
  │   │  │    REFERENCE CONTEXT:                                       │  │
  │   │  │    === Best Practice: REST Controller ===                   │  │
  │   │  │    @RestController                                          │  │
  │   │  │    @RequiredArgsConstructor                                 │  │
  │   │  │    // Best Practice: Constructor injection via Lombok...    │  │
  │   │  │    ...400 lines of real Java code examples...               │  │
  │   │  │                                                             │  │
  │   │  │    RULES: Include // Best Practice: comments...             │  │
  │   │  │  """                                                        │  │
  │   │                                                                │  │
  │   │  STEP 4: Stream from Ollama                                    │  │
  │   │  ├─ POST http://ollama:11434/api/chat (stream: true)          │  │
  │   │  │   messages: [                                               │  │
  │   │  │     {role: "system", content: <enriched prompt>},          │  │
  │   │  │     {role: "user", content: "Create a REST controller..."}  │  │
  │   │  │   ]                                                         │  │
  │   │  │                                                             │  │
  │   │  │  Ollama generates tokens with context awareness:            │  │
  │   │  │  "package com.example.api.controller;\n\n"                  │  │
  │   │  │  "// Best Practice: Use @RequiredArgsConstructor...\n"      │  │
  │   │  │  "@RestController\n"                                        │  │
  │   │  │   ... (follows the patterns from the injected snippets)     │  │
  │   │  │                                                             │  │
  │   │  STEP 5: Bridge SSE formats                                    │  │
  │   │  ├─ Ollama sends:  {"message":{"content":"@Rest"},"done":false}│  │
  │   │  └─ FastAPI emits: data: {"type":"content_block_delta",        │  │
  │   │                           "delta":{"text":"@Rest"}}            │  │
  │   └────────────────────────────────────────────────────────────────┘  │
  │                                                                      │
  ├─ For each SSE event:                                                 │
  │   ├─ Parse JSON                                                      │
  │   ├─ Append delta.text to generatedCode                              │
  │   └─ renderCode(generatedCode) → updates display with typewriter     │
  │                                                                      │
  └─ On [DONE]: remove cursor-blink, show "Done — 120 lines (RAG)"
```

### 4.4 Snippet Seeding into ChromaDB

```
FastAPI lifespan startup → seed_snippets(collection)

For each of the 10 snippets in spring_best_practices.json:
  │
  ├─ Read JSON entry:
  │   {id: "rest-controller-best-practice",
  │    category: "rest",
  │    title: "REST Controller Best Practice",
  │    description: "Production-grade REST controller...",
  │    code: "package com.example... @RestController..."}
  │
  ├─ Concatenate: title + "\n" + description + "\n\n" + code
  │   → "REST Controller Best Practice\nProduction-grade...\n\npackage com.example..."
  │
  ├─ Truncate to 2000 chars (for embedding model context limit)
  │
  ├─ POST http://ollama:11434/api/embed
  │   body: {model: "nomic-embed-text", input: <truncated text>}
  │   → Returns: [0.0234, -0.0891, 0.1456, ...0.0012] (768 floats)
  │
  └─ Collect: id, document, metadata, embedding

After all 10:
  │
  └─ collection.add(ids=[...], documents=[...], metadatas=[...], embeddings=[...])
     │
     └─ ChromaDB:
        ├─ Stores documents in SQLite (persistent)
        ├─ Builds HNSW index over 10 vectors × 768 dimensions
        └─ Index supports O(log n) approximate nearest-neighbor queries
```

---

## 5. Component Summary

### Ollama (LLM Server)

| Attribute | Value |
|-----------|-------|
| **Role** | LLM inference and text embedding |
| **Image** | `ollama/ollama:latest` |
| **Port** | 11434 |
| **Models** | `qwen2.5-coder:0.5b` (generation), `nomic-embed-text` (embedding) |
| **APIs used** | `/api/chat` (generation), `/api/embed` (embeddings), `/api/tags` (health), `/api/pull` (model download) |
| **Storage** | `ollama_data` volume (~350MB per model) |

### ChromaDB (Vector Database)

| Attribute | Value |
|-----------|-------|
| **Role** | Store and search code snippet embeddings |
| **Image** | `chromadb/chroma:latest` |
| **Port** | 8200 (host) → 8000 (container) |
| **Collection** | `spring_best_practices` (cosine distance metric) |
| **Index type** | HNSW (Hierarchical Navigable Small World) |
| **Documents** | 10 Java Spring best-practice code snippets |
| **Vector dimensions** | 768 (from nomic-embed-text) |
| **Storage** | `chroma_data` volume |

### LangChain RAG Service (FastAPI)

| Attribute | Value |
|-----------|-------|
| **Role** | RAG orchestration: embed → retrieve → augment → generate |
| **Image** | Custom (built from `langchain-service/Dockerfile`) |
| **Port** | 8100 |
| **Framework** | FastAPI + uvicorn |
| **Endpoints** | `/generate`, `/search`, `/snippets`, `/snippets/reload`, `/health` |
| **Key function** | `generate_code()` — the 3-step RAG pipeline |

### Spring Forge UI (nginx)

| Attribute | Value |
|-----------|-------|
| **Role** | Serve frontend + reverse proxy |
| **Image** | `nginx:alpine` |
| **Port** | 8080 (host) → 80 (container) |
| **Proxy routes** | `/ollama/*` → `ollama:11434`, `/rag/*` → `langchain-rag:8100` |
| **Frontend** | Single HTML file with Tailwind CSS + vanilla JavaScript |

---

## 6. Dependency Reference

### Docker Images

| Image | Version | Size | Purpose |
|-------|---------|------|---------|
| `ollama/ollama` | latest | ~1.2GB | LLM inference server |
| `chromadb/chroma` | latest | ~800MB | Vector database |
| `python:3.12-slim` | 3.12 | ~150MB | Base for RAG service |
| `nginx:alpine` | latest | ~5MB | Web server + reverse proxy |

### Python Packages (langchain-service)

| Package | Version | Purpose | Docs |
|---------|---------|---------|------|
| FastAPI | 0.115.6 | Async web framework | [fastapi.tiangolo.com](https://fastapi.tiangolo.com/) |
| uvicorn | 0.34.0 | ASGI server | [uvicorn.org](https://www.uvicorn.org/) |
| LangChain | 0.3.14 | LLM orchestration framework | [python.langchain.com](https://python.langchain.com/) |
| langchain-community | 0.3.14 | Community integrations | [python.langchain.com/docs/integrations](https://python.langchain.com/docs/integrations/) |
| langchain-chroma | 0.2.2 | ChromaDB integration | [python.langchain.com/docs/integrations/vectorstores/chroma](https://python.langchain.com/docs/integrations/vectorstores/chroma/) |
| langchain-ollama | 0.3.0 | Ollama integration | [python.langchain.com/docs/integrations/llms/ollama](https://python.langchain.com/docs/integrations/llms/ollama/) |
| chromadb-client | 0.5.23 | ChromaDB HTTP client | [docs.trychroma.com](https://docs.trychroma.com/) |
| httpx | 0.28.1 | Async HTTP client | [www.python-httpx.org](https://www.python-httpx.org/) |
| Pydantic | 2.10.4 | Data validation | [docs.pydantic.dev](https://docs.pydantic.dev/) |

### Frontend Libraries (CDN)

| Library | Version | Purpose | Docs |
|---------|---------|---------|------|
| Tailwind CSS | 3.x (CDN) | Utility-first CSS framework | [tailwindcss.com](https://tailwindcss.com/docs/) |
| JetBrains Mono | — | Monospace font for code | [jetbrains.com/lp/mono](https://www.jetbrains.com/lp/mono/) |
| DM Sans | — | UI body font | [fonts.google.com](https://fonts.google.com/specimen/DM+Sans) |

### Ollama Models

| Model | Parameters | Embedding Dims | Context | Purpose | Docs |
|-------|-----------|---------------|---------|---------|------|
| qwen2.5-coder:0.5b | 500M | — | 32K | Code generation | [ollama.com/library/qwen2.5-coder](https://ollama.com/library/qwen2.5-coder) |
| nomic-embed-text | 137M | 768 | 8K | Text embeddings | [ollama.com/library/nomic-embed-text](https://ollama.com/library/nomic-embed-text) |

---

## 7. Links and Further Reading

### Core Technologies

| Technology | Link | What to Read |
|------------|------|-------------|
| **Ollama** | [ollama.com](https://ollama.com/) | How to run LLMs locally |
| **Ollama API Reference** | [github.com/ollama/ollama/blob/main/docs/api.md](https://github.com/ollama/ollama/blob/main/docs/api.md) | `/api/chat`, `/api/embed`, `/api/pull` endpoint specs |
| **ChromaDB** | [docs.trychroma.com](https://docs.trychroma.com/) | Vector database concepts, collections, querying |
| **ChromaDB Python Client** | [docs.trychroma.com/docs/languages/python](https://docs.trychroma.com/docs/languages/python) | `collection.add()`, `collection.query()` API |
| **LangChain** | [python.langchain.com](https://python.langchain.com/) | RAG concepts, chains, embeddings, vector stores |
| **LangChain RAG Tutorial** | [python.langchain.com/docs/tutorials/rag](https://python.langchain.com/docs/tutorials/rag/) | Official RAG implementation guide |
| **FastAPI** | [fastapi.tiangolo.com](https://fastapi.tiangolo.com/) | Async endpoints, streaming responses, Pydantic models |
| **FastAPI Streaming** | [fastapi.tiangolo.com/advanced/custom-response/#streamingresponse](https://fastapi.tiangolo.com/advanced/custom-response/#streamingresponse) | `StreamingResponse` with async generators |

### Docker & Infrastructure

| Technology | Link | What to Read |
|------------|------|-------------|
| **Docker Compose** | [docs.docker.com/compose](https://docs.docker.com/compose/) | Service definitions, volumes, networking |
| **Docker Networking** | [docs.docker.com/network](https://docs.docker.com/network/) | How containers communicate by service name |
| **nginx Reverse Proxy** | [nginx.org/en/docs/http/ngx_http_proxy_module.html](https://nginx.org/en/docs/http/ngx_http_proxy_module.html) | `proxy_pass`, `proxy_buffering`, streaming config |
| **NVIDIA Container Toolkit** | [docs.nvidia.com/datacenter/cloud-native/container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) | GPU passthrough for Docker |

### Java Spring Best Practices

| Topic | Link |
|-------|------|
| **Spring Boot Reference** | [docs.spring.io/spring-boot/reference](https://docs.spring.io/spring-boot/reference/) |
| **Spring Data JPA** | [docs.spring.io/spring-data/jpa/reference/jpa.html](https://docs.spring.io/spring-data/jpa/reference/jpa.html) |
| **Spring Security** | [docs.spring.io/spring-security/reference](https://docs.spring.io/spring-security/reference/) |
| **MapStruct** | [mapstruct.org/documentation/stable/reference/html](https://mapstruct.org/documentation/stable/reference/html/) |
| **Testcontainers** | [testcontainers.com/guides/getting-started-with-testcontainers-for-java](https://testcontainers.com/guides/getting-started-with-testcontainers-for-java/) |
| **JUnit 5 User Guide** | [junit.org/junit5/docs/current/user-guide](https://junit.org/junit5/docs/current/user-guide/) |
| **Lombok** | [projectlombok.org/features](https://projectlombok.org/features/) |
| **OpenAPI / Swagger** | [springdoc.org](https://springdoc.org/) |

### Concepts

| Concept | Link | What to Read |
|---------|------|-------------|
| **RAG (Retrieval-Augmented Generation)** | [arxiv.org/abs/2005.11401](https://arxiv.org/abs/2005.11401) | Original RAG paper by Lewis et al. |
| **HNSW Algorithm** | [arxiv.org/abs/1603.09320](https://arxiv.org/abs/1603.09320) | How ChromaDB's vector search works |
| **Cosine Similarity** | [en.wikipedia.org/wiki/Cosine_similarity](https://en.wikipedia.org/wiki/Cosine_similarity) | Why we use cosine distance for text embeddings |
| **Server-Sent Events (SSE)** | [developer.mozilla.org/en-US/docs/Web/API/Server-sent_events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events) | The streaming protocol between backend and frontend |
| **12-Factor App** | [12factor.net](https://12factor.net/) | Why we use environment variables for config |




# Ollama + Claude Code — Local Offline Setup

Run [Claude Code](https://docs.claude.com/en/docs/claude-code/overview) entirely on your local machine using [Ollama](https://ollama.com) as the backend. No cloud API keys, no data leaving your machine.

## Prerequisites

- **Docker** & **Docker Compose** (v2)
- **Node.js 18+** (for Claude Code)
- **32GB+ RAM** recommended (16GB minimum with smaller models)
- **GPU** optional but strongly recommended (NVIDIA or AMD)

## Quick Start

```bash
# 1. Start everything (Ollama + Zactonics AI Spring Forge UI)
docker compose up -d

# 2. Pull a coding model
docker exec ollama ollama pull qwen3-coder

# 3. Open the Zactonics AI Spring Forge UI
#    → http://localhost:8080
```

The **Zactonics AI Spring Forge** web UI is a Java Spring-focused code generator that connects to your local Ollama instance. Features include dark/light mode, quick templates (REST controllers, JPA entities, services, etc.), streaming code output, and one-click download of `.java` files.

### Claude Code (Terminal) Setup

```bash
# 1. Install Claude Code (if not already installed)
npm install -g @anthropic-ai/claude-code

# 2. Run the setup script (starts Ollama + pulls default model)
chmod +x setup.sh
./setup.sh

# 3. Launch Claude Code
ANTHROPIC_BASE_URL=http://localhost:11434 \
ANTHROPIC_AUTH_TOKEN=ollama \
ANTHROPIC_API_KEY="" \
CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 \
claude --model qwen3-coder
```

## Manual Setup (Step by Step)

```bash
# Start Ollama
docker compose up -d

# Pull a model
docker exec ollama ollama pull qwen3-coder

# Configure Claude Code (add to ~/.bashrc or ~/.zshrc)
export ANTHROPIC_BASE_URL="http://localhost:11434"
export ANTHROPIC_AUTH_TOKEN="ollama"
export ANTHROPIC_API_KEY=""
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1

# Launch
claude --model qwen3-coder
```

## GPU Support

Edit `docker-compose.yml` and uncomment the appropriate section:

**NVIDIA GPU** — Requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html):
```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```

**AMD GPU (ROCm)**:
```yaml
image: ollama/ollama:rocm
devices:
  - /dev/kfd
  - /dev/dri
```

## Recommended Models

| Model | Size | Context | Best For |
|-------|------|---------|----------|
| `qwen3-coder` | ~30B MoE | 128K | General coding (default) |
| `glm-4.7-flash` | ~30B MoE (3B active) | 128K | Tool calling, agentic workflows |
| `nemotron-3-nano` | ~30B MoE | 128K | Coding + reasoning |
| `devstral` | ~24B | 128K | Mistral's coding model |
| `qwen2.5-coder:7b` | 7B | 32K | Low-RAM machines (16GB) |

Pull additional models anytime:
```bash
docker exec ollama ollama pull <model-name>
```

## Persistent Configuration

Instead of environment variables, add to `~/.claude/settings.json`:

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://localhost:11434",
    "ANTHROPIC_AUTH_TOKEN": "ollama",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"
  }
}
```

## Common Issues

| Problem | Solution |
|---------|----------|
| Connection refused | Make sure Ollama is running: `docker compose up -d` |
| Model not found | Check `docker exec ollama ollama list` for exact name |
| Timeout on first request | Pre-warm the model: `curl http://localhost:11434/api/generate -d '{"model":"qwen3-coder","prompt":"hi","stream":false}'` |
| Slow responses | Expected on CPU; use a GPU or smaller model |
| Tool calling not working | Use Ollama v0.14+ and a model that supports tool calling (glm-4.7-flash recommended) |

## Useful Commands

```bash
docker compose up -d          # Start Ollama
docker compose down            # Stop Ollama
docker compose logs -f ollama  # View logs
docker exec ollama ollama list # List installed models
docker exec ollama ollama rm <model>  # Remove a model
```

## Verify Offline

Disconnect from the internet and run a prompt. If you get a response, you're fully offline.
