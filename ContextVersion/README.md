# Zactonics AI Spring Forge — Local Offline Code Generator

Run AI-powered Java Spring code generation entirely on your local machine using [Ollama](https://ollama.com) as the LLM backend, **LangChain** for RAG orchestration, and **ChromaDB** as the vector store for best-practice code snippets.

## Architecture

```
┌──────────────────┐     ┌─────────────────────┐     ┌─────────────┐
│  Spring Forge UI │────▶│  LangChain RAG API  │────▶│  ChromaDB   │
│  (nginx :8080)   │     │  (FastAPI :8100)     │     │  (:8200)    │
└──────────────────┘     └──────────┬──────────┘     └─────────────┘
        │                           │
        │   Direct mode             │   RAG-enriched prompt
        ▼                           ▼
   ┌─────────────────────────────────────┐
   │          Ollama  (:11434)           │
   │   qwen2.5-coder / qwen3-coder      │
   │   nomic-embed-text (embeddings)     │
   └─────────────────────────────────────┘
```

**Two generation modes:**

| Mode | Flow | When to use |
|------|------|-------------|
| **Direct** (RAG Off) | UI → Ollama | Quick generation, simple prompts |
| **RAG** (RAG On) | UI → LangChain → ChromaDB → Ollama | Best-practice–aware generation with contextual examples |

## Prerequisites

- **Docker** & **Docker Compose** (v2)
- **Node.js 18+** (for Claude Code, optional)
- **32GB+ RAM** recommended (16GB minimum with smaller models)
- **GPU** optional but strongly recommended (NVIDIA or AMD)

## Quick Start

```bash
# 1. Start everything (Ollama + ChromaDB + LangChain RAG + Spring Forge UI)
docker compose up -d

# 2. Wait for the model-pull service to finish (~2-5 min first time)
docker compose logs -f model-pull

# 3. Open the Zactonics AI Spring Forge UI
#    → http://localhost:8080
#
# 4. Toggle "RAG On" in the header to enable best-practice context
```

### What happens on first start

1. **Ollama** starts and the `model-pull` service downloads the coding model + embedding model.
2. **ChromaDB** starts with persistent storage.
3. **LangChain RAG** service starts, waits for Ollama + ChromaDB, then:
   - Pulls the `nomic-embed-text` embedding model into Ollama
   - Seeds 10 Java Spring best-practice code snippets into ChromaDB
4. **Spring Forge UI** serves the web interface with nginx proxying all services.

### Service Ports

| Service | Port | URL |
|---------|------|-----|
| Spring Forge UI | 8080 | http://localhost:8080 |
| LangChain RAG API | 8100 | http://localhost:8100/health |
| ChromaDB | 8200 | http://localhost:8200 |
| Ollama | 11434 | http://localhost:11434 |

## Best-Practice Snippets (RAG Context)

When RAG mode is enabled, the LangChain service retrieves relevant Java Spring best-practice code snippets from ChromaDB and injects them into the LLM's system prompt. This teaches the model to follow your team's conventions.

### Included snippets (10 total):

| Snippet | Category | What it teaches |
|---------|----------|----------------|
| REST Controller | `rest` | ResponseEntity, @Valid, OpenAPI annotations, thin controllers |
| JPA Entity | `entity` | UUID keys, auditing, equals/hashCode, BigDecimal for money |
| Service Layer | `service` | @Transactional(readOnly), domain exceptions, SLF4J logging |
| Repository | `repo` | Custom @Query, projections, specifications, bulk updates |
| DTO + Mapper | `dto` | Create/Update/Response DTOs, MapStruct, @Valid nesting |
| Security Config | `security` | JWT filter chain, CORS, BCrypt, stateless sessions |
| Unit Tests | `test` | JUnit 5, Mockito BDD, @Nested, AssertJ assertions |
| Exception Handler | `exception` | @RestControllerAdvice, structured ErrorResponse, field errors |
| Application Config | `config` | application.yml, HikariCP, OSIV off, actuator, profiles |
| Integration Tests | `test` | @SpringBootTest, Testcontainers, TestRestTemplate |

### Adding your own snippets

Edit `langchain-service/snippets/spring_best_practices.json` and add entries following the same schema, then reload:

```bash
# Hot-reload snippets without restarting
curl -X POST http://localhost:8100/snippets/reload
```

## RAG API Endpoints

```bash
# Health check
curl http://localhost:8100/health

# Search for relevant snippets (without generating code)
curl -X POST "http://localhost:8100/search?prompt=REST+controller&top_k=3"

# List all loaded snippets
curl http://localhost:8100/snippets

# Generate code with RAG context
curl -X POST http://localhost:8100/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Create a REST controller for User CRUD", "model": "qwen2.5-coder:0.5b", "stream": false}'

# Force reload snippets from disk
curl -X POST http://localhost:8100/snippets/reload
```

## Claude Code (Terminal) Setup

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
| `devstral` | ~24B | 128K | Mistral's coding model |
| `qwen2.5-coder:7b` | 7B | 32K | Low-RAM machines (16GB) |
| `qwen2.5-coder:0.5b` | 0.5B | 32K | Ultra-low-RAM, fast iteration |

Pull additional models anytime:
```bash
docker exec ollama ollama pull <model-name>
```

## Useful Commands

```bash
docker compose up -d              # Start all services
docker compose down               # Stop all services
docker compose logs -f ollama     # View Ollama logs
docker compose logs -f langchain-rag  # View RAG service logs
docker compose logs -f chromadb   # View ChromaDB logs
docker exec ollama ollama list    # List installed models
docker exec ollama ollama rm <model>  # Remove a model
```

## Common Issues

| Problem | Solution |
|---------|----------|
| Connection refused | Make sure all services are running: `docker compose up -d` |
| RAG shows "offline" | Check LangChain logs: `docker compose logs langchain-rag` |
| ChromaDB not seeding | Ensure embedding model pulled: `docker exec ollama ollama pull nomic-embed-text` |
| Model not found | Check `docker exec ollama ollama list` for exact name |
| Timeout on first request | Models need warm-up; wait for `model-pull` to finish |
| Slow responses | Expected on CPU; use a GPU or smaller model |
