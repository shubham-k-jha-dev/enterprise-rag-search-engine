from backend.services.embedding import embed_text
from backend.services.vector_db import store_document
from backend.services.chunking import semantic_chunk

import os
from pypdf import PdfReader
import docx

def extract_text_from_txt(file_path):
    with open(file_path, "r", encoding = "utf-8") as f:
        return f.read()
    

def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    
    for page in reader.pages:
        text += page.extract_text()
        
    return text

def extract_text_from_docx(file_path):
    doc = docx.Document(file_path)
    text = ""
    
    for para in doc.paragraphs:
        text += para.text + "\n"
        
    return text

def process_document(file_path, workspace):
    filename = os.path.basename(file_path)
    
    if filename.endswith(".txt"):
        text = extract_text_from_txt(file_path)
        
    elif filename.endswith(".pdf"):
        text = extract_text_from_pdf(file_path)
        
    elif filename.endswith(".docx"):
        text = extract_text_from_docx(file_path)
        
    else:
        raise ValueError("Unsupported File Format")
    
    chunks = semantic_chunk(text)
    
    for i, chunk in enumerate(chunks):
        embedding = embed_text(chunk)
        
        metadata = {
            "text": chunk,
            "source": filename, 
            "chunk_id": i
        }
        
        store_document(embedding, metadata, workspace)
    
    return len(chunks)