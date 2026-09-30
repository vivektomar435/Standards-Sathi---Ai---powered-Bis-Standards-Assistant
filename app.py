from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from retrieval.hybrid import HybridRetriever
from llm.ollama import OllamaClient
from llm.prompts import build_messages


# ============================================================
# PATHS AND CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"

DB_PATH = str(BASE_DIR / "chroma_db")

# Current 1-standard collection.
# Change to "bis_standards" after the 10-standard ingestion.
COLLECTION_NAME = "bis_standards_test"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2:3b"

# Retrieve 5 candidates for retrieval accuracy.
TOP_K = 5
SEMANTIC_CANDIDATES = 15

# Send only the strongest 3 results to the LLM.
# This prevents large irrelevant chunks, such as Table 5,
# from unnecessarily slowing down Ollama.
LLM_CONTEXT_K = 3


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Standards Sathi API",
    version="1.0",
)


# ============================================================
# REQUEST MODEL
# ============================================================

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    history: list[dict] = Field(default_factory=list)
    lang: str = "English"


# ============================================================
# INITIALIZE RETRIEVER
# ============================================================

print("Loading BIS hybrid retriever...")

retriever = HybridRetriever(
    db_path=DB_PATH,
    collection_name=COLLECTION_NAME,
    embedding_model_name=EMBEDDING_MODEL,
    semantic_candidates=SEMANTIC_CANDIDATES,
)

print("Hybrid retriever loaded.")


# ============================================================
# INITIALIZE OLLAMA
# ============================================================

print("Connecting to Ollama...")

ollama = OllamaClient(
    base_url=OLLAMA_URL,
    model=OLLAMA_MODEL,
)

print("Standards Sathi backend ready.")


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/api/health")
def health():
    """
    Check whether the backend, Ollama server,
    and configured model are available.
    """

    available = ollama.is_available()

    model_available = False

    if available:
        try:
            model_available = ollama.model_available()
        except RuntimeError:
            model_available = False

    return {
        "status": "ok",
        "ollama_available": available,
        "model": OLLAMA_MODEL,
        "model_available": model_available,
        "collection": COLLECTION_NAME,
        "documents_in_collection": retriever.store.count(),
    }


# ============================================================
# CITATION BUILDER
# ============================================================

def citation(result):
    """
    Convert a retrieved result into the citation format
    expected by the Standards Sathi frontend.
    """

    metadata = result.get("metadata", {})

    standard = metadata.get("standard_number") or "Unknown Standard"

    if metadata.get("standard_year"):
        standard = f"{standard}:{metadata['standard_year']}"

    section = metadata.get("section_type")

    # --------------------------------------------------------
    # Clause
    # --------------------------------------------------------

    if section == "clause":

        if metadata.get("clause_number"):
            ref = f"Clause {metadata['clause_number']}"
        else:
            ref = "Clause"

        title = metadata.get("clause_title") or ref

    # --------------------------------------------------------
    # Table
    # --------------------------------------------------------

    elif section == "table":

        if metadata.get("table_number"):
            ref = f"Table {metadata['table_number']}"
        else:
            ref = "Table"

        title = ref

    # --------------------------------------------------------
    # Annex
    # --------------------------------------------------------

    elif section == "annex":

        if metadata.get("annex_number"):
            ref = f"Annex {metadata['annex_number']}"
        else:
            ref = "Annex"

        title = metadata.get("annex_title") or ref

    # --------------------------------------------------------
    # Annex Clause
    # --------------------------------------------------------

    elif section == "annex_clause":

        annex_number = metadata.get("annex_number")
        clause_number = metadata.get("clause_number")

        if annex_number and clause_number:
            ref = f"Annex {annex_number} Clause {clause_number}"
        elif annex_number:
            ref = f"Annex {annex_number}"
        else:
            ref = "Annex Clause"

        title = ref

    # --------------------------------------------------------
    # Other
    # --------------------------------------------------------

    else:

        ref = section or "Document"
        title = ref

    # --------------------------------------------------------
    # Page information
    # --------------------------------------------------------

    page_start = metadata.get("page_start")
    page_end = metadata.get("page_end")

    if page_start and page_end and page_end != page_start:

        page_text = f"Pages {page_start}-{page_end}"

    elif page_start:

        page_text = f"Page {page_start}"

    else:

        page_text = ""

    # --------------------------------------------------------
    # Final citation metadata
    # --------------------------------------------------------

    meta = " · ".join(
        item
        for item in [
            standard,
            ref,
            page_text,
        ]
        if item
    )

    return {
        "id": f"{standard} · {ref}",
        "title": title,
        "meta": meta,
    }


# ============================================================
# CHAT ENDPOINT
# ============================================================

@app.post("/api/chat")
def chat(request: ChatRequest):

    question = request.message.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty.",
        )

    # --------------------------------------------------------
    # Check Ollama server
    # --------------------------------------------------------

    if not ollama.is_available():

        raise HTTPException(
            status_code=503,
            detail="Ollama is not running. Start Ollama first.",
        )

    # --------------------------------------------------------
    # Check configured model
    # --------------------------------------------------------

    try:

        if not ollama.model_available():

            raise HTTPException(
                status_code=503,
                detail=(
                    f"Ollama model '{OLLAMA_MODEL}' "
                    "is not installed."
                ),
            )

    except RuntimeError as exc:

        raise HTTPException(
            status_code=503,
            detail=str(exc),
        )

    # --------------------------------------------------------
    # Retrieve relevant BIS chunks
    # --------------------------------------------------------

    results = retriever.search(
        question,
        top_k=TOP_K,
        candidate_k=SEMANTIC_CANDIDATES,
    )

    # --------------------------------------------------------
    # No retrieval result
    # --------------------------------------------------------

    if not results:

        return {
            "reply": (
                "The provided BIS context does not contain "
                "enough information to answer this question."
            ),
            "citations": [],
        }

    # --------------------------------------------------------
    # Reduce LLM context
    # --------------------------------------------------------
    #
    # We keep TOP_K=5 for retrieval quality.
    #
    # But we send only the strongest 3 results to Ollama.
    #
    # This is important because some chunks can be very large.
    # For example, Table 5 is around 4,268 characters and can
    # slow down Llama even when it is not needed to answer the
    # question.
    #
    # The waste-reduction question correctly retrieves Clause
    # 6.1 as rank 1, so this keeps the relevant evidence while
    # removing unnecessary lower-ranked context.
    # --------------------------------------------------------

    llm_results = results[:LLM_CONTEXT_K]

    # --------------------------------------------------------
    # Build grounded LLM messages
    # --------------------------------------------------------

    messages = build_messages(
        question=question,
        results=llm_results,
    )

    # --------------------------------------------------------
    # Generate answer with Ollama
    # --------------------------------------------------------

    try:

        answer = ollama.chat(
            messages=messages,
            temperature=0.0,
        )

    except RuntimeError as exc:

        raise HTTPException(
            status_code=502,
            detail=str(exc),
        )

    # --------------------------------------------------------
    # Empty answer protection
    # --------------------------------------------------------

    if not answer:

        raise HTTPException(
            status_code=502,
            detail="Ollama returned an empty answer.",
        )

    # --------------------------------------------------------
    # Return answer + citations
    # --------------------------------------------------------

    return {
        "reply": answer,
        "citations": [
            citation(result)
            for result in llm_results
        ],
    }


# ============================================================
# FRONTEND
# ============================================================

@app.get("/")
def frontend():

    index = FRONTEND_DIR / "index.html"

    if not index.exists():

        raise HTTPException(
            status_code=404,
            detail="frontend/index.html not found.",
        )

    return FileResponse(index)


# ============================================================
# API INFORMATION
# ============================================================

@app.get("/api")
def api_info():

    return {
        "name": "Standards Sathi",
        "status": "running",
        "docs": "/docs",
        "health": "/api/health",
    }