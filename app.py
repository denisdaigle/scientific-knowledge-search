import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
import os
from dotenv import load_dotenv
import json

load_dotenv()

# Initialize
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="text-embedding-3-small"
)

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection(
    name="scientific_papers",
    embedding_function=openai_ef
)

def extract_structured_data(text_chunk, query):
    """Extract structured fields using GPT-4"""
    
    prompt = f"""You are analyzing a scientific research document excerpt.

Query: {query}

Document excerpt:
{text_chunk[:2000]}

Extract the following if present in the text:
1. Research methodology/approach
2. Materials or substances studied
3. Key findings or outcomes
4. Challenges, problems, limitations, or failure modes mentioned (look for explicit statements AND implicit challenges being addressed)

Return as JSON with these exact keys:
{{"methodology": "...", "materials": "...", "findings": "...", "challenges": "..."}}

For challenges: Include both explicitly stated problems AND problems that are implicitly being solved by the research (e.g., if text discusses "stabilizing" something, the challenge is instability; if it discusses "improving biocompatibility", the challenge is poor biocompatibility).

If a field is not found, use "Not mentioned" as the value.
"""
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Cheap and fast
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=500
        )
        
        result = response.choices[0].message.content
        # Clean markdown if present
        if "```json" in result:
            result = result.split("```json")[1].split("```")[0]
        
        return json.loads(result)
    
    except Exception as e:
        return {"error": str(e)}

# Page config
st.set_page_config(
    page_title="Scientific Knowledge Search",
    page_icon="🔬",
    layout="wide"
)

# Header
st.title("🔬 Scientific Knowledge Search POC")
st.markdown("""
**Demo for Volta Effect R.26.01 - Knowledge Discovery Engine**

This prototype demonstrates semantic search on 100+ scientific papers. 
Search by concept, not just keywords.
""")

# Sidebar stats
with st.sidebar:
    st.header("📊 Corpus Stats")
    total_docs = collection.count()
    st.metric("Total Chunks", f"{total_docs:,}")
    st.metric("Source PDFs", "~30")
    
    st.markdown("---")
    st.markdown("**Try these queries:**")
    st.code("collagen crosslinking challenges")
    st.code("wet spinning failure modes")
    st.code("polymer degradation mechanisms")
    st.code("biocompatibility testing methods")

# Search interface
query = st.text_input(
    "🔍 Ask a conceptual question:",
    placeholder="e.g., What are common challenges with polymer crosslinking?"
)

num_results = st.slider("Number of results", 3, 10, 5)

if query:
    with st.spinner("Searching..."):
        # Semantic search
        results = collection.query(
            query_texts=[query],
            n_results=num_results
        )
        
        st.success(f"Found {len(results['documents'][0])} results")
        
        # Display results
        for idx, (doc, metadata, distance) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        )):
            with st.expander(f"**Result {idx+1}** - {metadata['source']} (Page {metadata['page']})", expanded=(idx==0)):
                # Relevance score
                relevance = 1 - distance  # Convert distance to similarity
                st.progress(relevance, text=f"Relevance: {relevance:.1%}")
                
                # Content
                st.markdown("**Content:**")
                st.text_area(
                    "Content",
                    doc,
                    height=200,
                    label_visibility="collapsed",
                    key=f"content_{idx}"
                )
                
                # Provenance
                st.markdown("**Source:**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("PDF", metadata['source'])
                with col2:
                    st.metric("Page", metadata['page'])
                with col3:
                    st.metric("Chunk", metadata.get('chunk', 0))
                
                # Extraction button
                st.markdown("---")
                if st.button(f"🔬 Extract Structured Data", key=f"extract_{idx}"):
                    with st.spinner("Extracting..."):
                        structured = extract_structured_data(doc, query)
                        
                        if "error" not in structured:
                            st.markdown("**Extracted Fields:**")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("**Methodology:**")
                                st.info(structured.get('methodology', 'N/A'))
                                
                                st.markdown("**Materials:**")
                                st.info(structured.get('materials', 'N/A'))
                            
                            with col2:
                                st.markdown("**Findings:**")
                                st.info(structured.get('findings', 'N/A'))
                                
                                st.markdown("**Challenges:**")
                                st.warning(structured.get('challenges', 'N/A'))
                        else:
                            st.error(f"Extraction failed: {structured['error']}")

# Footer
st.markdown("---")
st.markdown("""
**Built by:** Denis Daigle | Ridealong.co | KeepItCanadian.ai
**Tech Stack:** Streamlit + ChromaDB + OpenAI Embeddings  
**For:** Volta Effect R.26.01 Knowledge Discovery Engine POC
""")