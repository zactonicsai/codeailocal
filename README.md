# Ollama + Claude Code — Local Offline Setup

Run [Claude Code](https://docs.claude.com/en/docs/claude-code/overview) entirely on your local machine using [Ollama](https://ollama.com) as the backend. No cloud API keys, no data leaving your machine.

## Prerequisites

- **Docker** & **Docker Compose** (v2)
- **Node.js 18+** (for Claude Code)
- **32GB+ RAM** recommended (16GB minimum with smaller models)
- **GPU** optional but strongly recommended (NVIDIA or AMD)

## Quick Start

```bash
# 1. Install Claude Code (if not already installed)
npm install -g @anthropic-ai/claude-code

# 2. Run the setup script (starts Ollama + pulls default model)
chmod +x setup.sh
./setup.sh                     # defaults to qwen3-coder
# or
./setup.sh glm-4.7-flash      # pick a different model

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
