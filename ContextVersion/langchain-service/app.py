"""
LangChain RAG Service for Zactonics AI Spring Forge
====================================================
Provides context-enriched code generation by retrieving relevant Java Spring
best-practice snippets from ChromaDB and injecting them into the LLM prompt.

Architecture:
  [User Prompt] -> [FastAPI] -> [ChromaDB similarity search] -> [Build context]
       -> [Ollama LLM] -> [Streamed Java code response]

NOTE: This service uses ChromaDB's REST API directly via httpx instead of the
heavy 'chromadb' Python package. This avoids C compilation dependencies
(onnxruntime, numpy, etc.) and keeps the Docker image small and reliable.
"""

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# -- Configuration -------------------------------------------------------------

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
CHROMA_HOST = os.getenv("CHROMA_HOST", "chromadb")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
CHROMA_BASE_URL = f"http://{CHROMA_HOST}:{CHROMA_PORT}"
COLLECTION_NAME = "spring_best_practices"
SNIPPETS_PATH = Path("/app/snippets/spring_best_practices.json")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("langchain-rag")


# -- Thin ChromaDB REST Client ------------------------------------------------
# Replaces the heavy 'chromadb' Python package with direct HTTP calls.

class ChromaRESTClient:
    """Lightweight ChromaDB HTTP client using only httpx."""

    def __init__(self, base_url: str, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.collection_id: Optional[str] = None

    def _req(self, method: str, path: str, **kwargs) -> httpx.Response:
        kwargs.setdefault("timeout", self.timeout)
        url = f"{self.base_url}{path}"
        resp = httpx.request(method, url, **kwargs)
        resp.raise_for_status()
        return resp

    def heartbeat(self) -> dict:
        return self._req("GET", "/api/v1/heartbeat").json()

    def get_or_create_collection(self, name: str, metadata: Optional[dict] = None) -> str:
        body: dict = {"name": name, "get_or_create": True}
        if metadata:
            body["metadata"] = metadata
        data = self._req("POST", "/api/v1/collections", json=body).json()
        self.collection_id = data["id"]
        logger.info("Collection '%s' id=%s", name, self.collection_id)
        return self.collection_id

    def count(self) -> int:
        if not self.collection_id:
            return 0
        data = self._req("GET", f"/api/v1/collections/{self.collection_id}/count").json()
        return data if isinstance(data, int) else 0

    def add(self, ids: list, documents: list, metadatas: list, embeddings: list):
        self._req("POST", f"/api/v1/collections/{self.collection_id}/add", json={
            "ids": ids,
            "documents": documents,
            "metadatas": metadatas,
            "embeddings": embeddings,
        })

    def query(self, query_embeddings: list, n_results: int = 3,
              where: Optional[dict] = None, include: Optional[list] = None) -> dict:
        body: dict = {
            "query_embeddings": query_embeddings,
            "n_results": n_results,
            "include": include or ["documents", "metadatas", "distances"],
        }
        if where:
            body["where"] = where
        return self._req("POST", f"/api/v1/collections/{self.collection_id}/query", json=body).json()

    def get_all(self, include: Optional[list] = None) -> dict:
        body = {"include": include or ["metadatas"]}
        return self._req("POST", f"/api/v1/collections/{self.collection_id}/get", json=body).json()

    def delete_collection(self, name: str):
        self._req("DELETE", f"/api/v1/collections/{name}")
        self.collection_id = None


# -- Global State --------------------------------------------------------------

chroma: Optional[ChromaRESTClient] = None


# -- Helper Functions ----------------------------------------------------------

def wait_for_chroma(max_retries: int = 30, delay: float = 2.0) -> ChromaRESTClient:
    client = ChromaRESTClient(CHROMA_BASE_URL)
    for attempt in range(max_retries):
        try:
            client.heartbeat()
            logger.info("ChromaDB is ready (attempt %d)", attempt + 1)
            return client
        except Exception:
            logger.info("Waiting for ChromaDB... (attempt %d/%d)", attempt + 1, max_retries)
            time.sleep(delay)
    raise RuntimeError("ChromaDB not reachable after retries")


def wait_for_ollama(max_retries: int = 30, delay: float = 3.0):
    for attempt in range(max_retries):
        try:
            r = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
            if r.status_code == 200:
                logger.info("Ollama is ready (attempt %d)", attempt + 1)
                return
        except Exception:
            pass
        logger.info("Waiting for Ollama... (attempt %d/%d)", attempt + 1, max_retries)
        time.sleep(delay)
    raise RuntimeError("Ollama not reachable after retries")


def get_embedding(text: str) -> list:
    resp = httpx.post(
        f"{OLLAMA_BASE_URL}/api/embed",
        json={"model": EMBEDDING_MODEL, "input": text},
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["embeddings"][0]


def seed_snippets(client: ChromaRESTClient):
    existing = client.count()
    if existing > 0:
        logger.info("Collection already has %d documents, skipping seed.", existing)
        return

    if not SNIPPETS_PATH.exists():
        logger.warning("Snippets file not found at %s", SNIPPETS_PATH)
        return

    snippets = json.loads(SNIPPETS_PATH.read_text())
    logger.info("Seeding %d snippets into ChromaDB...", len(snippets))

    ids = []
    documents = []
    metadatas = []
    embeddings = []

    for s in snippets:
        doc_text = f"{s['title']}\n{s['description']}\n\n{s['code']}"
        ids.append(s["id"])
        documents.append(doc_text)
        metadatas.append({
            "category": s["category"],
            "title": s["title"],
            "description": s["description"],
        })
        emb = get_embedding(doc_text[:2000])
        embeddings.append(emb)

    client.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
    logger.info("Seeded %d snippets into collection '%s'", len(snippets), COLLECTION_NAME)


# -- Lifespan ------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global chroma

    logger.info("Starting RAG service...")
    wait_for_ollama()

    logger.info("Ensuring embedding model '%s' is available...", EMBEDDING_MODEL)
    try:
        httpx.post(
            f"{OLLAMA_BASE_URL}/api/pull",
            json={"name": EMBEDDING_MODEL, "stream": False},
            timeout=600,
        )
        logger.info("Embedding model ready")
    except Exception as e:
        logger.warning("Could not pull embedding model: %s", e)

    chroma = wait_for_chroma()
    chroma.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    seed_snippets(chroma)
    logger.info("RAG service ready")

    yield

    logger.info("Shutting down RAG service")


# -- FastAPI App ---------------------------------------------------------------

app = FastAPI(
    title="Spring Forge RAG API",
    description="RAG-powered Java Spring code generation with best-practice context",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# -- Request / Response Models -------------------------------------------------

class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="User's code generation prompt")
    model: str = Field(default="qwen2.5-coder:0.5b", description="Ollama model name")
    top_k: int = Field(default=3, ge=1, le=10, description="Number of snippets to retrieve")
    stream: bool = Field(default=True, description="Stream the response")
    category: Optional[str] = Field(default=None, description="Filter snippets by category")


class SnippetResult(BaseModel):
    id: str
    title: str
    category: str
    description: str
    score: float


class SearchResponse(BaseModel):
    query: str
    results: list[SnippetResult]


class HealthResponse(BaseModel):
    status: str
    chroma_connected: bool
    ollama_connected: bool
    snippet_count: int


# -- Endpoints -----------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
async def health():
    chroma_ok = False
    ollama_ok = False
    count = 0

    try:
        if chroma:
            chroma.heartbeat()
            chroma_ok = True
            count = chroma.count()
    except Exception:
        pass

    try:
        r = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        ollama_ok = r.status_code == 200
    except Exception:
        pass

    return HealthResponse(
        status="ok" if (chroma_ok and ollama_ok) else "degraded",
        chroma_connected=chroma_ok,
        ollama_connected=ollama_ok,
        snippet_count=count,
    )


@app.post("/search", response_model=SearchResponse)
async def search_snippets(prompt: str, top_k: int = 5, category: Optional[str] = None):
    if chroma is None:
        raise HTTPException(status_code=503, detail="ChromaDB not initialized")

    query_embedding = get_embedding(prompt)
    where_filter = {"category": category} if category else None

    results = chroma.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where_filter,
        include=["metadatas", "distances"],
    )

    snippets = []
    if results.get("ids") and results["ids"][0]:
        for i, doc_id in enumerate(results["ids"][0]):
            meta = results["metadatas"][0][i] if results.get("metadatas") else {}
            distance = results["distances"][0][i] if results.get("distances") else 0
            snippets.append(SnippetResult(
                id=doc_id,
                title=meta.get("title", ""),
                category=meta.get("category", ""),
                description=meta.get("description", ""),
                score=round(1 - distance, 4),
            ))

    return SearchResponse(query=prompt, results=snippets)


@app.post("/generate")
async def generate_code(req: GenerateRequest):
    if chroma is None:
        raise HTTPException(status_code=503, detail="ChromaDB not initialized")

    # Step 1: Retrieve relevant snippets
    query_embedding = get_embedding(req.prompt)
    where_filter = {"category": req.category} if req.category else None

    results = chroma.query(
        query_embeddings=[query_embedding],
        n_results=req.top_k,
        where=where_filter,
        include=["documents", "metadatas"],
    )

    # Build context block from retrieved snippets
    context_blocks = []
    if results.get("documents") and results["documents"][0]:
        for i, doc in enumerate(results["documents"][0]):
            meta = results["metadatas"][0][i] if results.get("metadatas") else {}
            context_blocks.append(
                f"=== Best Practice Reference: {meta.get('title', 'Snippet')} ===\n"
                f"Category: {meta.get('category', 'general')}\n"
                f"{meta.get('description', '')}\n\n"
                f"{doc}\n"
            )

    context = "\n".join(context_blocks)

    # Step 2: Build enriched prompt
    system_prompt = f"""You are a senior Java Spring Boot developer. You ONLY output clean, production-ready Java code.

REFERENCE CONTEXT - Use these best-practice examples as guidance for style, conventions, and patterns:

{context}

RULES:
- Output ONLY the Java source code, no markdown fences, no explanations outside the code.
- Use Java 17+ features where appropriate (records, sealed classes, pattern matching).
- Follow Spring Boot 3.x conventions.
- Include proper imports.
- Use Lombok where appropriate (@RequiredArgsConstructor, @Slf4j, @Data, @Builder).
- Add detailed Javadoc and inline // Best Practice: comments explaining WHY each decision was made.
- Follow the patterns shown in the reference context above.
- Apply SOLID principles and clean code conventions.

IMPORTANT: Include "// Best Practice:" comments throughout the code to explain design decisions,
just like the reference examples. These comments teach the developer WHY the code is written this way.

Do NOT wrap code in ```java or any markdown. Output raw Java code only."""

    # Step 3: Stream response from Ollama
    if req.stream:
        return StreamingResponse(
            _stream_ollama(req.model, system_prompt, req.prompt),
            media_type="text/event-stream",
        )
    else:
        return await _generate_ollama(req.model, system_prompt, req.prompt)


async def _stream_ollama(model: str, system_prompt: str, user_prompt: str):
    async with httpx.AsyncClient(timeout=httpx.Timeout(300)) as client:
        async with client.stream(
            "POST",
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": model,
                "stream": True,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
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


async def _generate_ollama(model: str, system_prompt: str, user_prompt: str) -> dict:
    async with httpx.AsyncClient(timeout=httpx.Timeout(300)) as client:
        response = await client.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": model,
                "stream": False,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            },
        )
        response.raise_for_status()
        data = response.json()
        return {
            "code": data.get("message", {}).get("content", ""),
            "model": model,
            "context_snippets_used": chroma.count() if chroma else 0,
        }


@app.get("/snippets")
async def list_snippets():
    if chroma is None:
        raise HTTPException(status_code=503, detail="ChromaDB not initialized")

    results = chroma.get_all(include=["metadatas"])
    items = []
    if results.get("ids"):
        for i in range(len(results["ids"])):
            meta = results["metadatas"][i] if results.get("metadatas") else {}
            items.append({"id": results["ids"][i], **meta})

    return {"count": len(items), "snippets": items}


@app.post("/snippets/reload")
async def reload_snippets():
    if chroma is None:
        raise HTTPException(status_code=503, detail="ChromaDB not initialized")

    try:
        chroma.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    chroma.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    seed_snippets(chroma)
    return {"status": "reloaded", "count": chroma.count()}
