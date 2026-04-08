import asyncio
import uuid
import logging
import logging.config
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles

# ── Logging setup ────────────────────────────────────────────────────────
logging.config.dictConfig({
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
            "datefmt": "%H:%M:%S",
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
        }
    },
    "loggers": {
        "logistics": {"handlers": ["console"], "level": "DEBUG", "propagate": False},
    },
    "root": {"handlers": ["console"], "level": "WARNING"},
})
log = logging.getLogger("logistics")

BASE_DIR = Path(__file__).parent          
ROOT_DIR = BASE_DIR.parent               

load_dotenv(dotenv_path=ROOT_DIR / ".env")  

from llm_service import LLMService
from models import (
    AskRequest,
    AskResponse,
    CollectionInfo,
    CollectionChunkInfo,
    DocumentListItem,
    ExtractRequest,
    ExtractResponse,
    ExtractionResult,
    ShipperConsignee,
    SourceChunk,
    UploadResponse,
)
from auth import authenticate_user, create_access_token, get_current_user
from processor import DocumentProcessor
from retriever import DocumentRetriever, compute_confidence

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.retriever = DocumentRetriever()   # loads embedding model + Chroma client
    app.state.processor = DocumentProcessor()
    app.state.llm = LLMService()
    yield


app = FastAPI(
    title="Logistics Document AI",
    description="Upload logistics documents and ask questions about them.",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/api-docs",    # move Swagger UI away from /docs
    redoc_url="/api-redoc",  # move ReDoc away too
)

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "logistics-document-ai"
    }


# ------------------------------------------------------------------
# Auth
# ------------------------------------------------------------------

@app.post("/auth/login")
async def login(form: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form.username, form.password)
    if not user:
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    token = create_access_token(user["username"])
    return {
        "access_token": token,
        "token_type":   "bearer",
        "display_name": user["display_name"],
    }


@app.get("/auth/me")
async def me(current_user: dict = Depends(get_current_user)):
    return {"username": current_user["username"], "display_name": current_user["display_name"]}


# ------------------------------------------------------------------
# Upload
# ------------------------------------------------------------------
# POST /upload          — parse, chunk, embed, index a document

@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...), current_user: dict = Depends(get_current_user)):
    """Parse, chunk, embed, and store a document in its own Chroma collection."""
    filename = file.filename or "unknown"
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    log.info("━" * 60)
    log.info(f"UPLOAD  {filename}  ({ext})")
    username=current_user["username"]

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    log.info(f"  File size : {len(data):,} bytes")

    processor: DocumentProcessor = app.state.processor
    retriever: DocumentRetriever = app.state.retriever

    try:
        full_text, page_texts = processor.parse(data, filename)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to parse document: {e}")

    log.info(f"  Pages     : {len(page_texts)}")
    log.info(f"  Full text : {len(full_text):,} chars")
    for pg in page_texts:
        log.debug(f"    Page {pg['page']:>3}  {len(pg['text']):>6,} chars")

    chunks = processor.chunk_text(page_texts)
    if not chunks:
        raise HTTPException(status_code=422, detail="No extractable text found in document.")

    # ── Chunk summary ────────────────────────────────────────────
    from collections import Counter
    type_counts = Counter(c["chunk_type"] for c in chunks)
    log.info(f"  Chunks    : {len(chunks)} total  —  " +
             "  ".join(f"{t}: {n}" for t, n in sorted(type_counts.items())))

    for i, c in enumerate(chunks):
        preview = c["text"][:120].replace("\n", " ↵ ")
        log.debug(
            f"    [{i:>3}] {c['chunk_type']:<10}  p{c['page_number']}  "
            f"{len(c['text']):>4} chars  │ {preview}"
        )

    doc_id = str(uuid.uuid4())
    log.info(f"  Indexing  → doc_id={doc_id[:8]}…")
    

    chunk_count = retriever.index_document(
        doc_id=doc_id,
        chunks=chunks,
        full_text=full_text,
        filename=filename,
        page_count=len(page_texts),username=username,
    )

    log.info(f"  Indexed   : {chunk_count} chunks embedded and stored in ChromaDB")
    log.info(f"  DONE  {filename}")
    log.info("━" * 60)

    return UploadResponse(
        doc_id=doc_id,
        filename=filename,
        chunk_count=chunk_count,
        page_count=len(page_texts),
    )


# ------------------------------------------------------------------
# Ask
# ------------------------------------------------------------------
#   POST /ask             — RAG Q&A with confidence score and guardrails

@app.post("/ask", response_model=AskResponse)
async def ask_question(req: AskRequest, current_user: dict = Depends(get_current_user)):
    """
    Hybrid RAG pipeline:
      1. HyDE   — LLM generates a hypothetical passage → embeds closer to doc text
      2. Expand — LLM generates 2 query variants → broader terminology coverage
      3. Hybrid — Dense (ChromaDB) + BM25 → RRF fusion → cross-encoder reranking
      4. Answer — LLM grounded strictly in top retrieved chunks
    """
    retriever: DocumentRetriever = app.state.retriever
    llm: LLMService = app.state.llm

    if not retriever.exists(req.doc_id, current_user["username"]):
        raise HTTPException(status_code=404, detail="Document not found. Upload it first.")

    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    log.info(f"ASK  doc={req.doc_id[:8]}…  q={req.question[:80]}")

    # ── Steps 1+2: HyDE and query expansion run concurrently ─────────────
    # Both are independent LLM calls (~0.8s each). Running sequentially wastes
    # ~0.8s on every query. asyncio.gather fires both simultaneously.
    log.debug("  [1+2/4] HyDE + query expansion (concurrent)…")
    hyde_text, expanded_queries = await asyncio.gather(
        asyncio.get_event_loop().run_in_executor(None, llm.generate_hyde, req.question),
        asyncio.get_event_loop().run_in_executor(None, llm.expand_query,  req.question),
    )

    # ── Step 3: Hybrid retrieval (Dense + BM25 + RRF + 3-large rerank)
    log.debug("  [3/4] Hybrid retrieval…")
    matched_chunks, similarities, guardrail_triggered = retriever.retrieve(
        doc_id=req.doc_id,
        username=current_user["username"],
        question=req.question,
        expanded_queries=expanded_queries,
        hyde_text=hyde_text,
    )

    if guardrail_triggered:
        best = float(similarities[0]) if len(similarities) > 0 else 0.0
        log.info(f"  Guardrail triggered  confidence={best:.3f}")
        return AskResponse(
            answer="Not found in document",
            confidence=round(max(best, 0.0), 4),
            source_chunks=[],
            guardrail_triggered=True,
        )

    # ── Step 4: Generate answer grounded in retrieved chunks
    log.debug(f"  [4/4] Generating answer from {len(matched_chunks)} chunks…")
    answer = llm.answer_question(req.question, matched_chunks, similarities)
    confidence = compute_confidence(similarities)

    log.info(f"  Done  confidence={confidence:.3f}  chunks_used={len(matched_chunks)}")

    source_chunks = [
        SourceChunk(
            text=c["text"],
            page_number=c["page_number"],
            similarity=round(float(similarities[i]) if i < len(similarities) else 0.0, 4),
            chunk_type=c["chunk_type"],
        )
        for i, c in enumerate(matched_chunks[:3])
    ]

    return AskResponse(
        answer=answer,
        confidence=confidence,
        source_chunks=source_chunks,
        guardrail_triggered=False,
    )


# ------------------------------------------------------------------
# Extract
# ------------------------------------------------------------------
#   POST /extract         — structured field extraction as JSON
@app.post("/extract", response_model=ExtractResponse)
async def extract_structured(req: ExtractRequest, _: dict = Depends(get_current_user)):
    """Extract structured shipment fields from an already-indexed document."""
    retriever: DocumentRetriever = app.state.retriever
    llm: LLMService = app.state.llm

    if not retriever.exists(req.doc_id):
        raise HTTPException(status_code=404, detail="Document not found. Upload it first.")

    full_text = retriever.get_full_text(req.doc_id) or ""
    return _run_extraction(full_text, llm, doc_id=req.doc_id)


# ------------------------------------------------------------------
# Direct extraction (no indexing — file uploaded just for extraction)
# ------------------------------------------------------------------

def _run_extraction(full_text: str, llm: LLMService, doc_id: str) -> ExtractResponse:
    """Shared helper used by both /extract and /extract-file."""
    try:
        raw = llm.extract_fields(full_text)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Extraction failed: {e}")

    for field in ("shipper", "consignee"):
        val = raw.get(field)
        if isinstance(val, dict):
            raw[field] = ShipperConsignee(**val)
        elif val is None:
            raw[field] = ShipperConsignee()

    known_fields = set(ExtractionResult.model_fields)
    result = ExtractionResult(**{k: v for k, v in raw.items() if k in known_fields})
    return ExtractResponse(doc_id=doc_id, data=result)


@app.post("/extract-file", response_model=ExtractResponse)
async def extract_from_file(file: UploadFile = File(...), _: dict = Depends(get_current_user)):
    """
    Extract structured shipment fields directly from an uploaded file.
    The file is parsed and extracted immediately — it is NOT indexed or
    stored in the vector database. Completely independent of the RAG pipeline.
    """
    filename = file.filename or "unknown"
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    processor: DocumentProcessor = app.state.processor
    llm: LLMService = app.state.llm

    try:
        full_text, _ = processor.parse(data, filename)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to parse document: {e}")

    if not full_text.strip():
        raise HTTPException(status_code=422, detail="No extractable text found in document.")

    return _run_extraction(full_text, llm, doc_id=filename)


# ------------------------------------------------------------------
# Collection management
# ------------------------------------------------------------------
#  GET  /docs            — list all indexed document collections
@app.get("/docs", response_model=list[DocumentListItem])
async def list_documents(current_user: dict = Depends(get_current_user)):
    """List all indexed document collections (persisted across restarts)."""
    retriever: DocumentRetriever = app.state.retriever
    return retriever.list_collections(current_user["username"])


@app.get("/docs/{doc_id}", response_model=CollectionInfo)
async def get_collection_info(doc_id: str, current_user: dict = Depends(get_current_user)):
    """Return full metadata and chunk listing for a specific document."""
    retriever: DocumentRetriever = app.state.retriever
    info = retriever.get_collection_info(doc_id, current_user["username"])
    if info is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    return info


@app.delete("/docs/{doc_id}")
async def delete_document(doc_id: str, current_user: dict = Depends(get_current_user)):
    """Permanently delete a document's Chroma collection."""
    retriever: DocumentRetriever = app.state.retriever
    deleted = retriever.delete(doc_id, current_user["username"])
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found.")
    return {"deleted": doc_id}




# ------------------------------------------------------------------
# Static UI (React build — only mounted when dist/ exists)
# In development the Vite dev server handles the frontend.
# In production (Docker) the React build is copied to backend/static/dist/
# ------------------------------------------------------------------

_STATIC_DIR = BASE_DIR / "static" / "dist"
if _STATIC_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(_STATIC_DIR / "assets")), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str):
        """Catch-all: serve index.html for all non-API routes (React SPA routing)."""
        return FileResponse(str(_STATIC_DIR / "index.html"))


# ------------------------------------------------------------------
# Dev entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, app_dir=str(BASE_DIR))
