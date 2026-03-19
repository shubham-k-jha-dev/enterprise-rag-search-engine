import os
from pypdf import PdfReader
import docx
import logging

from backend.services.embedding import embed_text
from backend.services.vector_db import store_document
from backend.services.chunking import semantic_chunk
from backend.services.logger import register_document_source

# Logger :- 
logger = logging.getLogger(__name__)

# TEXT EXTRACTION

def _extract_text_from_txt(file_path: str) -> str:
    """
    Extracts text from a .txt file.
    """
    with open(file_path, "r", encoding = "utf-8") as f:
        return f.read()
    

def _extract_text_from_pdf(file_path: str) -> str:
    """
    Extracts text from a .pdf file.
    """
    reader = PdfReader(file_path)
    text: str = ""
    
    for page in reader.pages:
        page_text: str = page.extract_text()
        if page_text:
            text += page_text
        
    return text

def _extract_text_from_docx(file_path: str) -> str:
    """
    Extracts and returns plain text from a .docx Word document.
    """
    doc = docx.Document(file_path)
    text: str = ""
 
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
 
    return text



# MAIN PROCESSING FUNCTION

def process_document(file_path: str, workspace: str) -> int:
    """
    Processes a document by extracting text, chunking it, embedding the chunks,
    and storing them in the vector database.
    """
    filename: str = os.path.basename(file_path)
    logger.info(
        "Processing document | filename='%s' | workspace='%s'",
        filename,
        workspace,
    )

    # Step 1 : Extract text
    if filename.endswith(".txt"):
        text = _extract_text_from_txt(file_path)
    elif filename.endswith(".pdf"):
        text = _extract_text_from_pdf(file_path)    
    elif filename.endswith(".docx"):
        text = _extract_text_from_docx(file_path)
 
    else:
        raise ValueError(
            f"Unsupported file format: '{filename}'. "
            "Supported formats: .txt, .pdf, .docx"
        )
    
    logger.info(
        "Text extracted | filename='%s' | length=%d chars",
        filename,
        len(text),
    )
    
    # Step 2 : Semantic Chunking
    chunks: list[str] = semantic_chunk(text)
    logger.info(
        "Chunking complete | filename='%s' | chunks=%d",
        filename,
        len(chunks),
    )
 
    # Step 3 + 4 : Embed and Store Each Chunk
    for i, chunk_text in enumerate(chunks):
        embedding: list[float] = embed_text(chunk_text)
 
        metadata: dict = {
            "text":     chunk_text,
            "source":   filename,
            "chunk_id": i,
        }
 
        store_document(embedding, metadata, workspace)
 
    logger.info(
        "All chunks indexed in Qdrant | filename='%s' | workspace='%s' | chunks=%d",
        filename,
        workspace,
        len(chunks),
    )
 
    #  Step 5: Register in SQLite (Phase D)
    register_document_source(filename, workspace)
    # Records (filename, workspace) in the document_sources SQLite table.
    # This is what enables keyword_search.py to find this file when
    # searching the "ca-finals" (or any) workspace — without relying
    # on the filename starting with the workspace name.
    
    return len(chunks)