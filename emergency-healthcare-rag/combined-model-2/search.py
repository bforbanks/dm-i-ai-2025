import os
import json
import pickle
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import re

# Configuration
EMBEDDINGS_FILE = "combined-model-2/embeddings.pkl"
SENTENCE_EMBEDDING_MODEL_PATH = "combined-model-2/local_sentence_model/"
SENTENCE_MODEL_NAME = "all-mpnet-base-v2"  # Upgraded from MiniLM
CHUNK_SIZE = 600  # Slightly larger chunks for better context
OVERLAP = 100  # More overlap for medical concepts

# Global model cache
_MODEL_CACHE = None

def setup_sentence_embedding_model():
    """Download and cache the sentence transformer model locally for offline use"""
    if not os.path.exists(SENTENCE_EMBEDDING_MODEL_PATH):
        print(f"Downloading sentence embedding model {SENTENCE_MODEL_NAME} for local caching...")
        # Download the model
        model = SentenceTransformer(SENTENCE_MODEL_NAME)
        # Save it locally
        os.makedirs(SENTENCE_EMBEDDING_MODEL_PATH, exist_ok=True)
        model.save(SENTENCE_EMBEDDING_MODEL_PATH)
        print(f"Sentence embedding model cached locally at {SENTENCE_EMBEDDING_MODEL_PATH}")
    else:
        print(f"Using cached sentence embedding model at {SENTENCE_EMBEDDING_MODEL_PATH}")

def load_sentence_embedding_model():
    """Load the locally cached sentence transformer model with caching"""
    global _MODEL_CACHE
    
    if _MODEL_CACHE is None:
        if not os.path.exists(SENTENCE_EMBEDDING_MODEL_PATH):
            print("Sentence embedding model not found. Setting up...")
            setup_sentence_embedding_model()
        
        print(f"Loading sentence embedding model from {SENTENCE_EMBEDDING_MODEL_PATH}")
        _MODEL_CACHE = SentenceTransformer(SENTENCE_EMBEDDING_MODEL_PATH)
    
    return _MODEL_CACHE

def load_all_documents() -> List[Tuple[str, str]]:
    """Load all markdown documents from condensed topics directory"""
    documents = []
    topics_dir = Path("data/condensed_topics")

    for topic_dir in topics_dir.iterdir():
        if topic_dir.is_dir():
            for md_file in topic_dir.glob("*.md"):
                with open(md_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                documents.append((str(md_file), content))
    
    return documents

def chunk_document_medical_aware(content: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> List[str]:
    """Split document into overlapping chunks with medical concept awareness"""
    # Split on medical headers and sections
    sections = re.split(r'\n##\s+', content)
    chunks = []
    
    for section in sections:
        if len(section.strip()) < 50:  # Skip very short sections
            continue
            
        # Split long sections into chunks
        words = section.split()
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if len(chunk.strip()) > 100:  # Only keep substantial chunks
                chunks.append(chunk.strip())
            
            if i + chunk_size >= len(words):
                break
                
    return chunks

def create_embeddings():
    """Generate embeddings for all document chunks AND topic names"""
    # Setup and use sentence embedding model
    setup_sentence_embedding_model()
    model = load_sentence_embedding_model()
    
    documents = load_all_documents()
    all_chunks = []
    chunk_metadata = []
    
    for file_path, content in documents:
        chunks = chunk_document_medical_aware(content)
        for chunk in chunks:
            all_chunks.append(chunk)
            chunk_metadata.append(file_path)
    
    print(f"Generating embeddings for {len(all_chunks)} chunks...")
    embeddings = model.encode(all_chunks)
    
    # Pre-compute topic embeddings to avoid recomputing them for every query
    topics_data = load_topics_mapping()
    topic_names = list(topics_data.keys())
    topic_embeddings = model.encode(topic_names)
    
    # Create BM25 index for keyword search
    tokenized_chunks = [chunk.lower().split() for chunk in all_chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    
    # Save everything
    data = {
        'embeddings': embeddings,
        'chunks': all_chunks,
        'metadata': chunk_metadata,
        'topic_embeddings': topic_embeddings,
        'topic_names': topic_names,
        'topics_data': topics_data,
        'bm25': bm25,
        'model_path': SENTENCE_EMBEDDING_MODEL_PATH
    }
    
    with open(EMBEDDINGS_FILE, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"Saved embeddings and BM25 index to {EMBEDDINGS_FILE}")

def load_embeddings():
    """Load pre-computed embeddings and BM25 index"""
    if not os.path.exists(EMBEDDINGS_FILE):
        print("Embeddings not found. Creating...")
        create_embeddings()
    
    with open(EMBEDDINGS_FILE, 'rb') as f:
        return pickle.load(f)

def load_topics_mapping():
    """Load topic name to ID mapping"""
    with open('data/topics.json', 'r') as f:
        return json.load(f)

def hybrid_search(statement: str, top_k: int = 10) -> List[Dict]:
    """
    Perform hybrid search combining BM25 and vector search
    """
    data = load_embeddings()
    model = load_sentence_embedding_model()
    
    # BM25 keyword search
    tokenized_query = statement.lower().split()
    bm25_scores = data['bm25'].get_scores(tokenized_query)
    bm25_indices = np.argsort(bm25_scores)[-top_k:][::-1]
    
    # Vector search
    query_embedding = model.encode([statement])
    similarities = np.dot(data['embeddings'], query_embedding.T).flatten()
    vector_indices = np.argsort(similarities)[-top_k:][::-1]
    
    # Reciprocal Rank Fusion (RRF)
    rrf_scores = {}
    k = 60  # RRF parameter
    
    # Score BM25 results
    for rank, idx in enumerate(bm25_indices):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (rank + k)
    
    # Score vector results
    for rank, idx in enumerate(vector_indices):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (rank + k)
    
    # Sort by combined scores
    sorted_indices = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Get topic information for top results
    results = []
    for idx, score in sorted_indices[:top_k]:
        chunk_text = data['chunks'][idx]
        file_path = data['metadata'][idx]
        
        # Extract topic from file path
        topic_name = Path(file_path).parent.name
        topic_id = data['topics_data'].get(topic_name, -1)
        
        results.append({
            'topic_id': topic_id,
            'topic_name': topic_name,
            'chunk_text': chunk_text,
            'file_path': file_path,
            'score': score
        })
    
    return results

def get_top_k_topics_with_context(statement: str, k: int = 8) -> List[Dict]:
    """
    Get top-K topics with context using hybrid search
    """
    hybrid_results = hybrid_search(statement, top_k=k*3)  # Get more for better deduplication
    
    # Deduplicate by topic and get best chunks for each topic
    topic_chunks = {}
    for result in hybrid_results:
        topic_id = result['topic_id']
        if topic_id not in topic_chunks or result['score'] > topic_chunks[topic_id]['score']:
            topic_chunks[topic_id] = result
    
    # Return top-K unique topics
    sorted_topics = sorted(topic_chunks.values(), key=lambda x: x['score'], reverse=True)
    return sorted_topics[:k]

def get_targeted_context_for_topic(statement: str, topic_id: int, max_chars: int = 2000) -> str:
    """
    Get targeted context for a specific topic
    """
    data = load_embeddings()
    model = load_sentence_embedding_model()
    
    # Find topic name from ID
    topic_name = None
    for name, tid in data['topics_data'].items():
        if tid == topic_id:
            topic_name = name
            break
    
    if not topic_name:
        return ""
    
    # Get chunks from this topic
    topic_chunks = []
    for i, file_path in enumerate(data['metadata']):
        if topic_name in file_path:
            topic_chunks.append({
                'text': data['chunks'][i],
                'index': i
            })
    
    # Re-rank chunks by relevance to statement
    statement_embedding = model.encode([statement])
    for chunk in topic_chunks:
        chunk_embedding = data['embeddings'][chunk['index']]
        similarity = np.dot(chunk_embedding, statement_embedding.T)[0]
        chunk['similarity'] = similarity
    
    # Sort by similarity and build context
    topic_chunks.sort(key=lambda x: x['similarity'], reverse=True)
    
    context = ""
    for chunk in topic_chunks:
        if len(context) + len(chunk['text']) < max_chars:
            context += chunk['text'] + "\n\n"
        else:
            break
    
    return context.strip() 