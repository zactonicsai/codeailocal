#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Ollama + Claude Code Local Setup
# Run Claude Code entirely offline with local models via Ollama
# ============================================================

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}╔══════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║   Ollama + Claude Code — Local Setup         ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════╝${NC}"
echo ""

# ----- Step 1: Start Ollama via Docker Compose -----
echo -e "${GREEN}[1/4] Starting Ollama via Docker Compose...${NC}"
docker compose up -d ollama
echo "  Waiting for Ollama to be ready..."
until curl -sf http://localhost:11434/api/tags > /dev/null 2>&1; do
  sleep 2
done
echo -e "  ${GREEN}✓ Ollama is running${NC}"
echo ""

# ----- Step 2: Pull a coding model -----
MODEL="${1:-qwen3-coder}"
echo -e "${GREEN}[2/4] Pulling model: ${MODEL}${NC}"
echo "  (This may take a while on first run depending on model size)"
docker exec ollama ollama pull "$MODEL"
echo -e "  ${GREEN}✓ Model ready${NC}"
echo ""

# ----- Step 3: Pre-warm the model -----
echo -e "${GREEN}[3/4] Pre-warming model (avoids cold-start timeout)...${NC}"
curl -sf http://localhost:11434/api/generate \
  -d "{\"model\": \"${MODEL}\", \"prompt\": \"test\", \"stream\": false}" > /dev/null 2>&1 || true
echo -e "  ${GREEN}✓ Model loaded into memory${NC}"
echo ""

# ----- Step 4: Check for Claude Code -----
echo -e "${GREEN}[4/4] Checking for Claude Code...${NC}"
if command -v claude &> /dev/null; then
  echo -e "  ${GREEN}✓ Claude Code is installed${NC}"
else
  echo -e "  ${YELLOW}⚠ Claude Code not found. Install it with:${NC}"
  echo "    npm install -g @anthropic-ai/claude-code"
fi
echo ""

# ----- Print launch instructions -----
echo -e "${CYAN}══════════════════════════════════════════════${NC}"
echo -e "${CYAN}  Setup complete! Launch Claude Code with:${NC}"
echo -e "${CYAN}══════════════════════════════════════════════${NC}"
echo ""
echo -e "  ${YELLOW}# Option A: Inline environment variables${NC}"
echo "  ANTHROPIC_BASE_URL=http://localhost:11434 \\"
echo "  ANTHROPIC_AUTH_TOKEN=ollama \\"
echo "  ANTHROPIC_API_KEY=\"\" \\"
echo "  CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 \\"
echo "  claude --model ${MODEL}"
echo ""
echo -e "  ${YELLOW}# Option B: Add to ~/.bashrc or ~/.zshrc for persistence${NC}"
echo "  export ANTHROPIC_BASE_URL=\"http://localhost:11434\""
echo "  export ANTHROPIC_AUTH_TOKEN=\"ollama\""
echo "  export ANTHROPIC_API_KEY=\"\""
echo "  export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1"
echo ""
echo -e "  ${YELLOW}# Option C: Add to ~/.claude/settings.json${NC}"
cat <<'JSON'
  {
    "env": {
      "ANTHROPIC_BASE_URL": "http://localhost:11434",
      "ANTHROPIC_AUTH_TOKEN": "ollama",
      "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"
    }
  }
JSON
echo ""
echo -e "  ${YELLOW}# Then just run:${NC}"
echo "  claude --model ${MODEL}"
echo ""
echo -e "${CYAN}Recommended models:${NC}"
echo "  qwen3-coder       — Great general coding (30B MoE, default)"
echo "  glm-4.7-flash     — Strong tool calling, 128K context"
echo "  nemotron-3-nano   — NVIDIA 30B MoE, good for coding"
echo "  devstral           — Mistral's coding model"
echo ""
echo -e "${CYAN}Tips:${NC}"
echo "  • 32GB+ RAM recommended for best experience"
echo "  • Use models with ≥64K context for Claude Code"
echo "  • To verify offline: disconnect internet and run a prompt"
echo "  • Stop Ollama: docker compose down"
echo "  • Logs: docker compose logs -f ollama"
