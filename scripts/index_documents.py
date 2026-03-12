import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.services.embedding import embed_text
from backend.services.vector_db import store_document, create_collection
from backend.services.chunking import semantic_chunk

def index_folder(folder_path):
    files = os.listdir(folder_path)
    
    txt_files = [f for f in files if f.endswith(".txt")]
    
    print(f"Found {len(txt_files)} text documents")
    
    for file in txt_files:
        filepath = os.path.join(folder_path, file)
        
        print(f"\n Indexing file: {file}")
        
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
            
        chunks = semantic_chunk(text)
        
        print(f"Created {len(chunks)} chunks")
        
        for i, chunk in enumerate(chunks):
            embedding = embed_text(chunk)
            
            metadata = {
                "text" : chunk,
                "source" : file, 
                "chunk_id" : i
            }
            
            store_document(embedding, metadata)
        
        print("File indexed successfully")
        
if __name__ == "__main__":
    if (len(sys.argv) < 2):
        print("Usage: python index_documents.py <folder_path>")
        sys.exit(1)
        
        
    create_collection()
    
    folder_path = sys.argv[1]
    
    index_folder(folder_path)