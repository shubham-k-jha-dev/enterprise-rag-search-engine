import logging
import os
import re
import shutil
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Query, Request, Security, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, field_validator
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIASGIMiddleware
from slowapi.util import get_remote_address

from backend.services.agentic_retrieval import run_agentic_retrieval
from backend.services.answer_generator import stream_answer
from backend.services.document_processor import process_document
from backend.services.keyword_search import rebuild_bm25
from backend.services.logger import (
    create_workspace,
    delete_workspace,
    get_all_workspaces,
    get_recent_logs,
    workspace_exists,
)
from backend.services.metrics import get_metrics
from backend.services.search import search
from backend.services.vector_db import create_collection, delete_collection

# Logging Setup :- 
logging.basicConfig(
    level = logging.INFO,
    format = "%(asctime)s - | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)


# Environment Variables :- 
load_dotenv()

# Constants :-
UPLOAD_FOLDER: str = "uploads"
DEFAULT_WORKSPACES: list[str] = ["default", "got", "dexter"]
ALLOWED_EXTENSIONS: set[str] = {".txt", ".pdf", ".docx"}
RATE_LIMIT: str = "30/minute"
os.makedirs(UPLOAD_FOLDER, exist_ok = True)

WORKSPACE_NAME_PATTERN = re.compile(
    r'^[a-z0-9][a-z0-9\-]{0,48}[a-z0-9]$|^[a-z0-9]$'
)


# Authentication :-
API_KEY: str | None = os.getenv("API_KEY")
api_key_header = APIKeyHeader(name = "X-API-KEY", auto_error = False)

def verify_api_key(api_key: str = Security(api_key_header)):
    # If key match, allow the request, if not, 403 Forbidden
    if not api_key or api_key != API_KEY:
        logger.warning("Rejected request - invalid or missing API Key.")
        raise HTTPException(
            status_code = 403,
            detail = "invalid or missing API Key. Pass it as X-API-Key header."
        )
    return api_key


# Rate Limiter :-
limiter = Limiter(key_func=get_remote_address)

# Life span :- 
@asynccontextmanager
async def lifespan(app: FastAPI):
    # On startup
    logger.info("Server starting — initialising Qdrant collections...")

    all_workspaces = get_all_workspaces()

    for ws in all_workspaces:
        create_collection(ws["name"])
        logger.info("Qdrant collection ready | workspace='%s'", ws["name"])

    logger.info("All workspaces initialised. Server is ready.")
    yield
    # On shutdown
    logger.info("Server shutting down.")
   
    
# App :-

app = FastAPI(
    title="AI Search Engine",
    description=(
        "Production-grade agentic retrieval system with hybrid search, "
        "BM25 keyword search, reciprocal rank fusion, cross-encoder reranking, "
        "Redis caching, and real-time LLM streaming via Groq."
    ),
    version="1.0.0",
    lifespan=lifespan
)

# Middleware registration

app.state.limiter = limiter
app.add_exception_handler(
    RateLimitExceeded, 
    lambda request, exc: JSONResponse(
        status_code = 429, 
        content={"error": f"Rate limit exceeded. Maximum: {RATE_LIMIT} per IP."}
    )
)
app.add_middleware(SlowAPIASGIMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        os.getenv("FRONTEND_ORIGIN", "http://localhost:8080")
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic Models :-
# define the exact shape of request and response bodies.
class SearchRequest(BaseModel):
    query: str
    workspace: str = "default"
    source: str | None = None
    
class SearchResult(BaseModel):
    text: str
    source: str | None = None
    chunk_id: int | None = None
    rerank_score: float | None = None
    
class SearchResponse(BaseModel):
    query: str
    workspace: str
    results: list[SearchResult]

class WorkspaceCreateRequest(BaseModel):
    """
    Request body for POST /workspace.
    """
    name: str
    description: str = ""

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        """
        Validates workspace name againt WORKSPACE_NAME_PATTERN
        """
        if not WORKSPACE_NAME_PATTERN.match(value):
            raise ValueError(
                "Workspace name must be 1-50 characters, "
                "lowercase letters/digits/hyphens only, "
                "must start and end with a letter or digit. "
                f"Got: '{value}'"
            )
        return value
    

# Endpoints :- 

@app.get("/health")
def health() -> dict:
    # Used by docker to know everything is ok
    return {"status": "ok", "service": "ai-search-engine"}


@app.get("/workspaces", tags=["Workspaces"])
def list_workspaces() -> dict:
    # Returns available workspaces from SQLite. Used for frontend dropdowns
    all_ws = get_all_workspaces()
    names: list[str] = [ws["name"] for ws in all_ws]
    return {"workspaces": names}


@app.post("/workspaces", tags=["Workspaces"], status_code=201)
def create_workspace_api(
    request: WorkspaceCreateRequest,
    api_key: str = Security(verify_api_key),
) -> dict:
    """
    Creates a new workspace.
    """
    logger.info("Create workspace request | name='%s'", request.name)

    if workspace_exists(request.name):
        raise HTTPException(
            status_code=409,
            detail=f"Workspace '{request.name}' already exists.",
        )

    create_collection(request.name)

    success: bool = create_workspace(request.name, request.description)

    if not success:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to register workspace '{request.name}'.",
        )

    logger.info("Workspace created | name='%s'", request.name)

    return {
        "message": f"Workspace '{request.name}' created successfully.",
        "workspace": request.name,
        "description": request.description,
    }


@app.delete("/workspaces/{name}", tags=["Workspaces"])
def delete_workspace_api(
    name: str,
    api_key: str = Security(verify_api_key),
) -> dict:
    """
    Deletes a workspace and its Qdrant vector collection.
    Default workspaces (default, got, dexter) are protected.
    """
    logger.info("Delete workspace request | name='%s'", name)

    if name in DEFAULT_WORKSPACES:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot delete default workspace '{name}'.",
        )

    if not workspace_exists(name):
        raise HTTPException(
            status_code=404,
            detail=f"Workspace '{name}' not found.",
        )

    delete_collection(name)

    deleted: bool = delete_workspace(name)

    if not deleted:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete workspace '{name}' from registry.",
        )

    rebuild_bm25()

    logger.info("Workspace deleted | name='%s'", name)

    return {"message": f"Workspace '{name}' deleted successfully."}



@app.post("/search", response_model=SearchResponse, tags=["Retrieval"])
def search_api(
    request: SearchRequest,
    api_key: str = Security(verify_api_key),
) -> SearchResponse:
    logger.info(
        "Search request | workspace='%s' | query='%s'",
        request.workspace,
        request.query,
    )

    if not workspace_exists(request.workspace):
        raise HTTPException(
            status_code=404,
            detail=f"Workspace '{request.workspace}' not found.",
        )

    try:
        results = search(
            request.query,
            workspace=request.workspace,
            source=request.source,
        )
        return SearchResponse(
            query=request.query,
            workspace=request.workspace,
            results=[
                SearchResult(**r) if isinstance(r, dict) else SearchResult(text=r)
                for r in results
            ],
        )
    except Exception as e:
        logger.error("Search failed | error='%s'", e)
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
    
    
@app.get("/ask", tags = ["Generation"])
@limiter.limit(RATE_LIMIT)
def ask_api(
    request: Request, 
    query: str, 
    workspace: str = "default",
    conversation_id: str = "default",
    api_key: str = Security(verify_api_key)
) -> StreamingResponse:
    """
    Agentic retrieval + streaming LLM answer generation with conversation memory.
    """
    logger.info(
        "Ask request | workspace='%s' | conversation_id='%s' | query='%s'",
        workspace,
        conversation_id,
        query,
    )

    if not workspace_exists(workspace):
        raise HTTPException(
            status_code=404,
            detail=f"Workspace '{workspace}' not found.",
        )

    try:
        docs = run_agentic_retrieval(query=query, workspace=workspace)
        return StreamingResponse(
            stream_answer(query, docs, conversation_id, workspace),
            media_type="text/plain",
        )
    except Exception as e:
        logger.error(
            "Ask failed | workspace='%s' | conversation_id='%s' | error='%s'",
            workspace,
            conversation_id,
            e,
        )
        raise HTTPException(status_code = 500, detail = f"Answer generation failed: {str(e)}")
    
    
@app.get("/metrics", tags = ["System"])
def metrics_api() -> dict:
    return get_metrics()


@app.get("/logs", tags = ["System"])
def logs_api(api_key: str = Security(verify_api_key)):
    return {"logs": get_recent_logs(limit=20)}


@app.post("/upload")
async def upload_documents(
    file: UploadFile = File(...),
    workspace: str = Query(...),
    api_key: str = Security(verify_api_key),    
) -> dict:  
    logger.info(
        "Upload request | workspace='%s' | filename='%s'",
        workspace,
        file.filename,
    )
    # Step 1: Validate file extension
    ext = os.path.splitext(file.filename)[1].lower()
    
    if ext not in ALLOWED_EXTENSIONS:
        logger.warning(
            "Upload rejected — unsupported extension '%s' for file '%s'",
            ext,
            file.filename,
        )
        raise HTTPException(
            status_code=400,
            # 400 Bad Request — the client sent invalid data.
            detail=(
                f"Unsupported file type '{ext}'. "
                f"Allowed types: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
            ),
        )

    if not workspace_exists(workspace):
        raise HTTPException(
            status_code=404,
            detail=(
                f"Workspace '{workspace}' not found. "
                "Create it first via POST /workspaces."
            ),
        )

    # Step 2: Save file to disk
    file_path: str = os.path.join(UPLOAD_FOLDER, file.filename)
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info("File saved to disk: '%s'", file_path)

    # Step 3 - 6: Process document
        chunks_created: int = process_document(file_path, workspace)
        logger.info(
            "Document indexed | file = '%s' | workspace = '%s' | chunks = %d",
            file.filename,
            workspace,
            chunks_created,
        )

    # Step 7: Rebuild BM25 Index
        rebuild_bm25()
        
        return {
            "message": "Document uploaded and indexed successfully",
            "file": file.filename,
            "workspace": workspace,
            "chunks_created": chunks_created
        }
    except Exception as e:
        logger.error(
                "Upload failed | file='%s' | workspace='%s' | error='%s'",
                file.filename,
                workspace,
                e,
            )
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")