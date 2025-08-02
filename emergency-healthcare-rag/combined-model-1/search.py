import os
import json
import pickle
from pathlib import Path
from typing import List, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer

EMBEDDINGS_FILE = "combined-model-1/embeddings.pkl"
SENTENCE_EMBEDDING_MODEL_PATH = "combined-model-1/local_sentence_model/"
SENTENCE_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500
OVERLAP = 50

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
    """Load the locally cached sentence transformer model"""
    if not os.path.exists(SENTENCE_EMBEDDING_MODEL_PATH):
        print("Sentence embedding model not found. Setting up...")
        setup_sentence_embedding_model()
    
    print(f"Loading sentence embedding model from {SENTENCE_EMBEDDING_MODEL_PATH}")
    return SentenceTransformer(SENTENCE_EMBEDDING_MODEL_PATH)

def load_sentence_embedding_model_fallback():
    """Load sentence embedding model with fallback to old path"""
    if os.path.exists(SENTENCE_EMBEDDING_MODEL_PATH):
        return SentenceTransformer(SENTENCE_EMBEDDING_MODEL_PATH)
    elif os.path.exists("../model-1/local_sentence_model/"):
        print("Using fallback sentence embedding model path")
        return SentenceTransformer("../model-1/local_sentence_model/")
    else:
        print("No sentence embedding model found, setting up...")
        setup_sentence_embedding_model()
        return SentenceTransformer(SENTENCE_EMBEDDING_MODEL_PATH)

def load_all_documents() -> List[Tuple[str, str]]:
    """Load all markdown documents from topics directory"""
    documents = []
    # Change this path to use different topic sources:
    #   "data/topics"           - original full articles
    #   "data/condensed_topics" - cleaned versions
    topics_dir = Path("data/condensed_topics")

    for topic_dir in topics_dir.iterdir():
        if topic_dir.is_dir():
            for md_file in topic_dir.glob("*.md"):
                with open(md_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                documents.append((str(md_file), content))
    
    return documents

def chunk_document(content: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> List[str]:
    """Split document into overlapping chunks"""
    words = content.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if len(chunk.strip()) > 0:
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
        chunks = chunk_document(content)
        for chunk in chunks:
            all_chunks.append(chunk)
            chunk_metadata.append(file_path)
    
    print(f"Generating embeddings for {len(all_chunks)} chunks...")
    embeddings = model.encode(all_chunks)
    
    # Pre-compute topic embeddings to avoid recomputing them for every query
    print("Generating embeddings for topic names...")
    with open('data/topics.json', 'r') as f:
        topics = json.load(f)
    topic_names = list(topics.keys())
    topic_embeddings = model.encode(topic_names)
    
    data = {
        'embeddings': embeddings,
        'chunks': all_chunks,
        'metadata': chunk_metadata,
        'model_path': SENTENCE_EMBEDDING_MODEL_PATH,  # Store sentence embedding model path
        'topic_names': topic_names,
        'topic_embeddings': topic_embeddings,
        'topics_mapping': topics
    }
    
    with open(EMBEDDINGS_FILE, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"Saved embeddings to {EMBEDDINGS_FILE}")

def load_embeddings():
    """Load pre-computed embeddings"""
    if not os.path.exists(EMBEDDINGS_FILE):
        raise FileNotFoundError(f"Embeddings file {EMBEDDINGS_FILE} not found. Run create_embeddings() first.")
    
    with open(EMBEDDINGS_FILE, 'rb') as f:
        return pickle.load(f)

def search_relevant_content(statement: str, top_k: int = 1, max_chars: int = 1500) -> str:
    """Search for minimal relevant content for truth classification"""
    data = load_embeddings()
    # Use sentence embedding model path if available, fallback to model name for old embeddings
    if 'model_path' in data:
        # Check if the stored path exists, otherwise use fallback
        stored_path = data['model_path']
        if os.path.exists(stored_path):
            model = SentenceTransformer(stored_path)
        else:
            # Use fallback to old path
            model = load_sentence_embedding_model_fallback()
    else:
        # Fallback for old embeddings files
        model = load_sentence_embedding_model_fallback()
    
    query_embedding = model.encode([statement])
    
    similarities = np.dot(query_embedding, data['embeddings'].T).flatten()
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    relevant_chunks = []
    total_chars = 0
    for idx in top_indices:
        chunk = data['chunks'][idx]
        if total_chars + len(chunk) <= max_chars:
            relevant_chunks.append(chunk)
            total_chars += len(chunk)
        else:
            remaining_chars = max_chars - total_chars
            if remaining_chars > 100:
                relevant_chunks.append(chunk[:remaining_chars] + "...")
            break
    
    return "\n\n".join(relevant_chunks)

def get_top_k_topics_with_context_from_embedding(statement_embedding: np.ndarray, k: int = 3) -> list:
    """Get top K most likely topics with their relevant context chunks using pre-computed embedding"""
    data = load_embeddings()
    
    # OPTIMIZATION: Use pre-computed topic embeddings instead of recomputing
    topic_names = data['topic_names']
    topic_embeddings = data['topic_embeddings']
    topics = data['topics_mapping']
    
    # Method 1: Topic name similarity using pre-computed embeddings
    similarities = np.dot(statement_embedding, topic_embeddings.T).flatten()
    top_k_indices = np.argsort(similarities)[-k:][::-1]
    
    # Method 2: Also find top relevant chunks and group by topic
    # OPTIMIZATION: Reuse the same statement embedding
    chunk_similarities = np.dot(statement_embedding, data['embeddings'].T).flatten()
    top_chunk_indices = np.argsort(chunk_similarities)[-20:][::-1]  # Get top 20 chunks
    
    # Group chunks by topic based on their content
    topic_chunk_scores = {}
    for chunk_idx in top_chunk_indices:
        chunk_text = data['chunks'][chunk_idx]
        chunk_embedding = data['embeddings'][chunk_idx:chunk_idx+1]
        
        # Find which topic this chunk best matches
        chunk_topic_similarities = np.dot(chunk_embedding, topic_embeddings.T).flatten()
        best_topic_idx = np.argmax(chunk_topic_similarities)
        best_topic_name = topic_names[best_topic_idx]
        topic_id = topics[best_topic_name]
        
        if topic_id not in topic_chunk_scores:
            topic_chunk_scores[topic_id] = {
                'name': best_topic_name,
                'chunks': [],
                'total_score': 0,
                'name_similarity': 0
            }
        
        topic_chunk_scores[topic_id]['chunks'].append({
            'text': chunk_text,
            'score': chunk_similarities[chunk_idx]
        })
        topic_chunk_scores[topic_id]['total_score'] += chunk_similarities[chunk_idx]
    
    # Add name similarities for top K topics
    for idx in top_k_indices:
        topic_name = topic_names[idx]
        topic_id = topics[topic_name]
        if topic_id not in topic_chunk_scores:
            topic_chunk_scores[topic_id] = {
                'name': topic_name,
                'chunks': [],
                'total_score': 0,
                'name_similarity': similarities[idx]
            }
        else:
            topic_chunk_scores[topic_id]['name_similarity'] = similarities[idx]
    
    # Combine scores: name similarity + chunk relevance
    final_topic_scores = []
    for topic_id, data in topic_chunk_scores.items():
        combined_score = data['name_similarity'] * 0.4 + data['total_score'] * 0.6
        final_topic_scores.append({
            'topic_id': topic_id,
            'topic_name': data['name'],
            'score': combined_score,
            'name_similarity': data['name_similarity'],
            'chunk_score': data['total_score'],
            'best_chunks': sorted(data['chunks'], key=lambda x: x['score'], reverse=True)[:2]
        })
    
    # Return top K topics sorted by combined score
    return sorted(final_topic_scores, key=lambda x: x['score'], reverse=True)[:k]

def get_top_k_topics_with_context(statement: str, k: int = 3) -> list:
    """Get top K most likely topics with their relevant context chunks (OPTIMIZED)"""
    data = load_embeddings()
    # Use sentence embedding model path if available, fallback to model name for old embeddings
    if 'model_path' in data:
        # Check if the stored path exists, otherwise use fallback
        stored_path = data['model_path']
        if os.path.exists(stored_path):
            model = SentenceTransformer(stored_path)
        else:
            # Use fallback to old path
            model = load_sentence_embedding_model_fallback()
    else:
        # Fallback for old embeddings files
        model = load_sentence_embedding_model_fallback()
    
    # OPTIMIZATION: Embed statement only once
    statement_embedding = model.encode([statement])
    
    # Use the new function with pre-computed embedding
    return get_top_k_topics_with_context_from_embedding(statement_embedding, k)

def identify_topic_from_embeddings(statement: str) -> int:
    """Use improved semantic search to identify the most likely topic"""
    top_topics = get_top_k_topics_with_context(statement, k=3)
    return top_topics[0]['topic_id'] if top_topics else 0

def get_targeted_context_for_topic(statement: str, topic_id: int, max_chars: int = 1500) -> str:
    """Get context chunks specifically relevant to the chosen topic"""
    top_topics = get_top_k_topics_with_context(statement, k=3)
    
    # Find our target topic in the results
    target_topic = None
    for topic in top_topics:
        if topic['topic_id'] == topic_id:
            target_topic = topic
            break
    
    if not target_topic or not target_topic['best_chunks']:
        # Fallback to general search if no specific chunks found
        return search_relevant_content(statement, top_k=1, max_chars=max_chars)
    
    # Use the best chunks for this specific topic
    context_chunks = []
    total_chars = 0
    
    for chunk in target_topic['best_chunks']:
        chunk_text = chunk['text']
        if total_chars + len(chunk_text) <= max_chars:
            context_chunks.append(chunk_text)
            total_chars += len(chunk_text)
        else:
            # Add partial chunk if there's space
            remaining_chars = max_chars - total_chars
            if remaining_chars > 100:
                context_chunks.append(chunk_text[:remaining_chars] + "...")
            break
    
    return "\n\n".join(context_chunks)

def get_multi_topic_context(candidate_topics: list, max_chars: int = 1500) -> str:
    """Get context from multiple candidate topics for LLM selection"""
    context_chunks = []
    total_chars = 0
    chars_per_topic = max_chars // len(candidate_topics)  # Distribute evenly
    
    for topic in candidate_topics:
        topic_chunks = []
        topic_chars = 0
        
        for chunk in topic['best_chunks']:
            chunk_text = chunk['text']
            if topic_chars + len(chunk_text) <= chars_per_topic:
                topic_chunks.append(chunk_text)
                topic_chars += len(chunk_text)
            else:
                # Add partial chunk if there's space
                remaining_chars = chars_per_topic - topic_chars
                if remaining_chars > 100:
                    topic_chunks.append(chunk_text[:remaining_chars] + "...")
                break
        
        if topic_chunks:
            topic_context = f"=== {topic['topic_name']} ===\n" + "\n".join(topic_chunks)
            if total_chars + len(topic_context) <= max_chars:
                context_chunks.append(topic_context)
                total_chars += len(topic_context)
    
    return "\n\n".join(context_chunks)