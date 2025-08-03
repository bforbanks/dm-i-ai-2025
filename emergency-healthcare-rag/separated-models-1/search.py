#!/usr/bin/env python3
"""
Search module for separated-models-1
Improved hybrid search with rich context retrieval
"""

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
EMBEDDINGS_FILE = "separated-models-1/embeddings.pkl"
SENTENCE_EMBEDDING_MODEL_PATH = "separated-models-1/local_sentence_model/"
SENTENCE_MODEL_NAME = "all-mpnet-base-v2"
CHUNK_SIZE = 500  # Smaller chunks for more granular context
OVERLAP = 150     # More overlap for better context continuity

# Global model cache
_MODEL_CACHE = None

def setup_sentence_embedding_model():
    """Download and cache the sentence transformer model locally"""
    if not os.path.exists(SENTENCE_EMBEDDING_MODEL_PATH):
        print(f"Downloading sentence embedding model {SENTENCE_MODEL_NAME} for local caching...")
        model = SentenceTransformer(SENTENCE_MODEL_NAME)
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
            
        # Split long sections into chunks with more overlap
        words = section.split()
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if len(chunk.strip()) > 100:  # Only keep substantial chunks
                chunks.append(chunk.strip())
            
            if i + chunk_size >= len(words):
                break
                
    return chunks

def create_embeddings():
    """Generate embeddings for all document chunks"""
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
    
    # Create BM25 index for keyword search
    tokenized_chunks = [chunk.lower().split() for chunk in all_chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    
    # Load topics mapping
    topics_data = load_topics_mapping()
    
    # Save everything
    data = {
        'embeddings': embeddings,
        'chunks': all_chunks,
        'metadata': chunk_metadata,
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

def hybrid_search(statement: str, top_k: int = 20) -> List[Dict]:
    """
    Perform hybrid search combining BM25 and vector search
    Returns individual chunks, not deduplicated by topic
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
    
    # Get chunk information for top results
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

def get_best_topic(statement: str) -> int:
    """
    Determine the best topic using hybrid search
    Returns the topic ID of the highest-scoring chunk
    """
    search_results = hybrid_search(statement, top_k=10)
    
    if not search_results:
        return 0  # Default fallback
    
    # Return the topic of the highest-scoring result
    return search_results[0]['topic_id']

def get_rich_context_for_statement(statement: str, topic_id: int, max_chars: int = 4000) -> str:
    """
    Get rich context using only the full documents from the chosen topic
    """
    # Find topic name from ID
    topic_name = None
    topics_data = load_topics_mapping()
    for name, tid in topics_data.items():
        if tid == topic_id:
            topic_name = name
            break
    
    if not topic_name:
        return ""
    
    # Get the FULL document for the chosen topic only
    full_document = get_full_document_for_topic(topic_name)
    
    # Build context with only the topic's full document
    context = f"MEDICAL CONTEXT FOR TOPIC: {topic_name}\n\n"
    context += full_document
    
    # Truncate if too long
    if len(context) > max_chars:
        context = context[:max_chars] + "..."
    
    return context.strip()

def get_full_document_for_topic(topic_name: str) -> str:
    """
    Get the full document content for a specific topic
    """
    topics_dir = Path("data/condensed_topics")
    topic_dir = topics_dir / topic_name
    
    if not topic_dir.exists():
        return ""
    
    # Combine all markdown files for this topic
    full_content = ""
    for md_file in topic_dir.glob("*.md"):
        with open(md_file, 'r', encoding='utf-8') as f:
            content = f.read()
            full_content += f"\n\n--- {md_file.name} ---\n\n"
            full_content += content
    
    return full_content.strip() 