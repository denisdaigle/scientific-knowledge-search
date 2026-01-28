import os
import fitz  # pymupdf
from pathlib import Path
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
import json

load_dotenv()

# Initialize OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Create collection with OpenAI embeddings
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="text-embedding-3-small"
)

collection = chroma_client.get_or_create_collection(
    name="scientific_papers",
    embedding_function=openai_ef
)

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF with page numbers"""
    doc = fitz.open(pdf_path)
    pages_text = []
    
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text()
        if text.strip():  # Only include pages with text
            pages_text.append({
                'page_num': page_num,
                'text': text.strip()
            })
    
    doc.close()
    return pages_text

def chunk_text(text, chunk_size=1000, overlap=200):
    """Split text into overlapping chunks"""
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    
    return chunks

def ingest_pdfs(pdf_dir="data/pdfs"):
    """Ingest all PDFs into ChromaDB"""
    pdf_dir = Path(pdf_dir)
    pdf_files = list(pdf_dir.glob("*.pdf"))
    
    print(f"Found {len(pdf_files)} PDFs to ingest")
    
    documents = []
    metadatas = []
    ids = []
    
    for idx, pdf_path in enumerate(pdf_files):
        print(f"\nProcessing {idx+1}/{len(pdf_files)}: {pdf_path.name}")
        
        try:
            pages = extract_text_from_pdf(pdf_path)
            
            for page_data in pages:
                # Chunk the page text
                chunks = chunk_text(page_data['text'])
                
                for chunk_idx, chunk in enumerate(chunks):
                    doc_id = f"{pdf_path.stem}_p{page_data['page_num']}_c{chunk_idx}"
                    
                    documents.append(chunk)
                    metadatas.append({
                        'source': pdf_path.name,
                        'page': page_data['page_num'],
                        'chunk': chunk_idx
                    })
                    ids.append(doc_id)
            
            print(f"  ✓ Extracted {len(pages)} pages")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            continue
    
    # Add to ChromaDB in batches
    batch_size = 100
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i+batch_size]
        batch_meta = metadatas[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]
        
        collection.add(
            documents=batch_docs,
            metadatas=batch_meta,
            ids=batch_ids
        )
        print(f"Added batch {i//batch_size + 1}")
    
    print(f"\n✓ Ingestion complete! Total chunks: {len(documents)}")

if __name__ == "__main__":
    ingest_pdfs()

