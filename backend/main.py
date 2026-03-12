from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIASGIMiddleware
import shutil
import os
from fastapi import UploadFile, File, Query, FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from contextlib import asynccontextmanager
from backend.services.vector_db import create_collection
from backend.services.search import search
from backend.services.answer_generator import stream_answer
from backend.services.metrics import get_metrics
from backend.services.document_processor import process_document
from backend.services.agentic_retrieval import run_agentic_retrieval
from backend.services.keyword_search import rebuild_bm25


# Constants :-
UPLOAD_FOLDER = "uploads"
DEFAULT_WORKSPACES = ["default", "got", "dexter"]
os.makedirs(UPLOAD_FOLDER, exist_ok = True)

# Rate Limiter :-
limiter = Limiter(key_func=get_remote_address)

@asynccontextmanager
async def lifespan(app: FastAPI):
    for workspace in DEFAULT_WORKSPACES:
        create_collection(workspace)
    yield   
    
# App :-
app = FastAPI(
    title="AI Search Engine",
    description="Production-grade agentic retrieval system with hybrid search and reranking",
    version="1.0.0",
    lifespan=lifespan
)

app.state.limiter = limiter
app.add_exception_handler(
    RateLimitExceeded, 
    lambda request, exc: JSONResponse(
        status_code = 429, 
        content = {"error" : "Too many requests. Limit: 30/minute"}
    )
)
app.add_middleware(SlowAPIASGIMiddleware)

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
    

# Endpoints :- 

@app.get("/health")
def health():
    # Used by docker to know everything is ok
    return {"status": "ok", "service": "ai-search-engine"}


@app.get("/workspaces")
def list_workspaces():
    # Returns available workspaces. Used for frontend dropdowns
    return {"workspaces": DEFAULT_WORKSPACES}



@app.post("/search", response_model=SearchResponse)
def search_api(request: SearchRequest):
    try:
        results = search(request.query, workspace = request.workspace, source = request.source)
        return SearchResponse(
            query=request.query,
            workspace=request.workspace,
            results=[SearchResult(**r) if isinstance(r, dict) else SearchResult(text=r) for r in results]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
    
    
@app.get("/ask")
@limiter.limit("30/minute")
def ask_api(request: Request, query: str, workspace: str = "default"):
    # Streaming endpoint 
    try:
        docs = run_agentic_retrieval(query = query, workspace = workspace)
        return StreamingResponse(
            stream_answer(query, docs),
            media_type="text/plain"
        )
    except Exception as e:
        raise HTTPException(status_code = 500, detail = f"Answer generation failed: {str(e)}")
    
    
@app.get("/metrics")
def metrics_api():
    return get_metrics()


@app.post("/upload")
async def upload_documents(
    file: UploadFile = File(...),
    workspace: str = Query(...),    
):  
    # validate file type - txt, pdf, docx
    allowed_extensions = {".txt", ".pdf", ".docx"}
    ext = os.path.splitext(file.filename)[1].lower()
    
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code = 400, 
            detail = f"Unsupported file type '{ext}. Allowed: {allowed_extensions}"
        )
        
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Process document
        chunks_created = process_document(file_path, workspace)
        
        rebuild_bm25()
        
        return {
            "message": "Document uploaded and indexed successfully",
            "file": file.filename,
            "workspace": workspace,
            "chunks_created": chunks_created
        }
    except Excpetion as e:
        raise HTTPException(status_code = 500, detail = f"Upload failed: {str(e)}")