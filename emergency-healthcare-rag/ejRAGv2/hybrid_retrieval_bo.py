"""
Hybrid Retrieval BO Optimization: BM25 + Embedding (MiniLM or ColBERT)

This script performs Bayesian optimization over a hybrid retrieval system,
combining BM25 and dense embedding-based similarity (MiniLM or ColBERT),
using a weighted linear combination of their scores.

Hyperparameters searched:
- BM25: chunk_size_bm25, overlap_bm25, k1, b
- Embedding: model_type (MiniLM, ColBERT), chunk_size_embed, overlap_embed
- Weights: alpha (BM25), beta (embed model)

Objective: Maximize top-1 accuracy across validation statements.
"""

import os
import json
import pickle
from pathlib import Path
from typing import Literal, List, Dict, Tuple, Optional, Any, Callable
import numpy as np
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util
from bayes_opt import BayesianOptimization
import pandas as pd
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# ---------------------------------------------------------------------------
# Paths and Constants
# ---------------------------------------------------------------------------
TOPIC_DIR = Path("data/topics")
STATEMENT_DIR = Path("data/train/statements")
ANSWER_DIR = Path("data/train/answers")
TOPIC_MAP = json.loads(Path("data/topics.json").read_text())

# Models
MINILM_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
# Note: ColBERT model is commented out as it requires specific installation
# COLBERT_MODEL = SentenceTransformer("colbert-ir/colbertv2.0")

# Cache directory
CACHE_DIR = Path(".cache")
CACHE_DIR.mkdir(exist_ok=True)

# Results storage
RESULTS_FILE = Path(".cache/bo_results.json")
RESULTS_CSV_FILE = Path(".cache/bo_results.csv")
# Incremental snapshot JSON (updated after every trial)
RESULTS_SNAPSHOT_FILE = Path(".cache/bo_results_incremental.json")

# Numerical threshold to consider embedding weight effectively zero
EMBEDDING_WEIGHT_EPS = 1e-8

# In-memory caches removed per user request (avoid caching chunks/embeddings across trials)

# Optional callback to report phase progress during a single evaluation
EVAL_PROGRESS_CALLBACK: Optional[Callable[[int], None]] = None

# ---------------------------------------------------------------------------
# Data Loading Functions
# ---------------------------------------------------------------------------
def load_statements():
    """Load validation statements and their true topics."""
    records = []
    for path in sorted(STATEMENT_DIR.glob("*.txt")):
        sid = path.stem.split("_")[1]
        statement = path.read_text().strip()
        ans = json.loads((ANSWER_DIR / f"statement_{sid}.json").read_text())
        records.append((statement, ans["statement_topic"]))
    return records

def load_topic_docs() -> Dict[int, str]:
    """Load all topic documents."""
    topic_docs = {}
    for topic_path in TOPIC_DIR.iterdir():
        topic_id = TOPIC_MAP[topic_path.name]
        contents = []
        for md_file in topic_path.glob("*.md"):
            text = md_file.read_text(encoding="utf-8").strip()
            contents.append(text)
        topic_docs[topic_id] = "\n".join(contents)
    return topic_docs

# ---------------------------------------------------------------------------
# Chunking Functions
# ---------------------------------------------------------------------------
def chunk_words(words: List[str], size: int, overlap: int):
    """Yield word windows of length *size* with given *overlap*."""
    step = max(1, size - overlap)
    for i in range(0, len(words), step):
        yield words[i : i + size]
        if i + size >= len(words):
            break

def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Chunk text into overlapping segments."""
    words = text.split()
    chunks = []
    for w_chunk in chunk_words(words, chunk_size, overlap):
        if len(w_chunk) < 10:  # skip very small fragments
            continue
        chunk_text = " ".join(w_chunk)
        chunks.append(chunk_text)
    return chunks

# ---------------------------------------------------------------------------
# BM25 Indexing and Search
# ---------------------------------------------------------------------------
def build_bm25_index(chunk_size: int, overlap: int, k1: float, b: float) -> Dict:
    """Build BM25 index with specified parameters."""
    chunk_size = int(chunk_size)
    overlap = int(overlap)
    # Use exact floating point values for k1 and b (no rounding)
    k1 = float(k1)
    b = float(b)
    
    chunks: List[str] = []
    topics: List[int] = []
    chunk_texts: List[str] = []

    # Get all markdown files first
    md_files = list(TOPIC_DIR.rglob("*.md"))
    
    for md_file in tqdm(md_files, desc="Processing BM25 documents", unit="file"):
        topic_name = md_file.parent.name
        topic_id = TOPIC_MAP[topic_name]
        text = md_file.read_text(encoding="utf-8").strip()
        text_chunks = chunk_text(text, chunk_size, overlap)
        
        for chunk in text_chunks:
            chunks.append(chunk)
            topics.append(topic_id)
            chunk_texts.append(chunk)

    # Build BM25 tokenized corpus
    print(f"Tokenizing BM25 chunks ({len(chunk_texts)} chunks)...")
    tokenized_chunks = [chunk.lower().split() for chunk in chunk_texts]

    # Instantiate BM25 with custom k1 and b
    print(f"Building BM25 index with {len(chunk_texts)} chunks (k1={k1}, b={b})...")
    bm25 = BM25Okapi(tokenized_chunks, k1=k1, b=b)

    return {
        'topics': topics,
        'chunk_texts': chunk_texts,
        'bm25': bm25,
        'chunk_size': chunk_size,
        'overlap': overlap,
        'k1': k1,
        'b': b
    }

def bm25_search(query: str, bm25_data: Dict, top_k: int = 10) -> List[Tuple[int, float]]:
    """Perform BM25 search and return top-k results with topic IDs and scores."""
    tokenized_query = query.lower().split()
    scores = bm25_data['bm25'].get_scores(tokenized_query)
    
    # Get top-k indices
    top_indices = np.argsort(scores)[-top_k:][::-1]
    
    results = []
    for idx in top_indices:
        topic_id = bm25_data['topics'][idx]
        score = scores[idx]
        results.append((topic_id, score))
    
    return results

# ---------------------------------------------------------------------------
# Embedding Indexing and Search
# ---------------------------------------------------------------------------
def build_embedding_index(chunk_size: int, overlap: int, model_type: str) -> Dict:
    """Build embedding index with specified parameters."""
    chunk_size = int(chunk_size)
    overlap = int(overlap)
    
    # Select model
    if model_type == "MiniLM":
        model = MINILM_MODEL
    elif model_type == "ColBERT":
        # For now, use MiniLM as ColBERT placeholder
        # model = COLBERT_MODEL
        model = MINILM_MODEL
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    chunks: List[str] = []
    topics: List[int] = []
    chunk_texts: List[str] = []

    # Get all markdown files first
    md_files = list(TOPIC_DIR.rglob("*.md"))
    
    for md_file in tqdm(md_files, desc="Processing embedding documents", unit="file"):
        topic_name = md_file.parent.name
        topic_id = TOPIC_MAP[topic_name]
        text = md_file.read_text(encoding="utf-8").strip()
        text_chunks = chunk_text(text, chunk_size, overlap)
        
        for chunk in text_chunks:
            chunks.append(chunk)
            topics.append(topic_id)
            chunk_texts.append(chunk)

    # Encode all chunks
    print(f"Encoding {len(chunk_texts)} chunks with {model_type}...")
    embeddings = model.encode(chunk_texts, convert_to_tensor=True, show_progress_bar=True)

    result = {
        'topics': topics,
        'chunk_texts': chunk_texts,
        'embeddings': embeddings,
        'model': model,
        'chunk_size': chunk_size,
        'overlap': overlap,
        'model_type': model_type
    }
    return result

def embedding_search(query: str, embed_data: Dict, top_k: int = 10) -> List[Tuple[int, float]]:
    """Perform embedding search and return top-k results with topic IDs and scores."""
    query_embedding = embed_data['model'].encode(query, convert_to_tensor=True)
    scores = util.cos_sim(query_embedding, embed_data['embeddings'])[0]
    
    top_k_indices = np.argsort(-scores.cpu().numpy())
    results = []
    for i in top_k_indices[:top_k]:
        topic_id = embed_data['topics'][i]
        score = scores[i].item()
        results.append((topic_id, score))
    
    return results

# ---------------------------------------------------------------------------
# Hybrid Retrieval
# ---------------------------------------------------------------------------
def hybrid_search(
    query: str, 
    bm25_data: Dict, 
    embed_data: Optional[Dict], 
    alpha: float, 
    beta: float, 
    top_k: int = 10
) -> List[Tuple[int, float]]:
    """Perform hybrid search combining BM25 and embedding scores."""
    # Get individual results
    bm25_results = bm25_search(query, bm25_data, top_k=top_k*2)  # Get more for better fusion
    embed_results: List[Tuple[int, float]] = []
    if embed_data is not None and beta > EMBEDDING_WEIGHT_EPS:
        embed_results = embedding_search(query, embed_data, top_k=top_k*2)
    
    # Create score dictionaries
    bm25_scores = {topic_id: score for topic_id, score in bm25_results}
    embed_scores = {topic_id: score for topic_id, score in embed_results}
    
    # Normalize scores to [0, 1] range
    if bm25_scores:
        bm25_max = max(bm25_scores.values())
        bm25_min = min(bm25_scores.values())
        if bm25_max > bm25_min:
            bm25_scores = {k: (v - bm25_min) / (bm25_max - bm25_min) for k, v in bm25_scores.items()}
    
    if embed_scores:
        embed_max = max(embed_scores.values())
        embed_min = min(embed_scores.values())
        if embed_max > embed_min:
            embed_scores = {k: (v - embed_min) / (embed_max - embed_min) for k, v in embed_scores.items()}
    
    # Combine scores
    hybrid_scores = {}
    all_topics = set(bm25_scores.keys()) | set(embed_scores.keys())
    
    for topic_id in all_topics:
        bm25_score = bm25_scores.get(topic_id, 0.0)
        embed_score = embed_scores.get(topic_id, 0.0)
        hybrid_score = alpha * bm25_score + beta * embed_score
        hybrid_scores[topic_id] = hybrid_score
    
    # Sort by hybrid score
    sorted_results = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_results[:top_k]

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_config(
    chunk_size_bm25: int,
    overlap_bm25: int,
    k1: float,
    b: float,
    chunk_size_embed: int,
    overlap_embed: int,
    model_type: str,
    alpha: float,
    beta: float
) -> float:
    """Evaluate a configuration and return top-1 accuracy."""
    try:
        # Build indices
        print(f"Building BM25 index (chunk={chunk_size_bm25}, overlap={overlap_bm25}, k1={k1}, b={b})...")
        bm25_data = build_bm25_index(chunk_size_bm25, overlap_bm25, k1, b)
        
        # Only build embedding index if beta is effectively non-zero
        if beta > EMBEDDING_WEIGHT_EPS:
            print(f"Building embedding index (chunk={chunk_size_embed}, overlap={overlap_embed}, model={model_type})...")
            embed_data = build_embedding_index(chunk_size_embed, overlap_embed, model_type)
        else:
            print("Skipping embedding index build (beta≈0)")
            embed_data = None
        
        # Load validation data
        print("Loading validation statements...")
        statements = load_statements()

        # Indices built once per trial and reused within evaluation only
        
        # Evaluate with progress bar
        correct = 0
        total = len(statements)
        
        print(f"Evaluating {total} statements with α={alpha:.3f}, β={beta:.3f}...")
        for idx, (query, true_topic) in enumerate(tqdm(statements, desc="Evaluating", unit="query")):
            results = hybrid_search(query, bm25_data, embed_data, alpha, beta, top_k=1)
            if results and results[0][0] == true_topic:
                correct += 1
            if EVAL_PROGRESS_CALLBACK is not None:
                # Report per-statement progress to the outer loop if requested
                try:
                    EVAL_PROGRESS_CALLBACK(idx + 1)
                except Exception:
                    pass
        
        accuracy = correct / total if total > 0 else 0.0
        print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")
        return accuracy
        
    except Exception as e:
        print(f"Error in evaluation: {e}")
        return 0.0

# ---------------------------------------------------------------------------
# Bayesian Optimization
# ---------------------------------------------------------------------------
def objective_function(
    chunk_size_bm25,
    overlap_bm25,
    k1,
    b,
    chunk_size_embed,
    overlap_embed,
    alpha,
    beta,
    model_selector  # 0 = MiniLM, 1 = ColBERT
):
    """Objective function for Bayesian optimization."""
    # Cast inputs to usable types
    cs_bm25 = int(round(chunk_size_bm25))
    ov_bm25 = int(round(overlap_bm25))
    cs_embed = int(round(chunk_size_embed))
    ov_embed = int(round(overlap_embed))
    k1 = float(k1)
    b = float(b)
    alpha = float(alpha)
    beta = float(beta)

    # Safety clamps to prevent pathological chunking (e.g., overlap >= chunk_size)
    cs_bm25 = max(2, cs_bm25)
    cs_embed = max(2, cs_embed)
    ov_bm25 = max(0, min(ov_bm25, cs_bm25 - 1))
    ov_embed = max(0, min(ov_embed, cs_embed - 1))

    model_type = "MiniLM" if model_selector < 0.5 else "ColBERT"

    # Evaluate configuration
    accuracy = evaluate_config(
        cs_bm25, ov_bm25, k1, b,
        cs_embed, ov_embed, model_type,
        alpha, beta
    )

    return accuracy

## removed auxiliary re-evaluation helper for alpha/beta sprays

def _clip_to_bounds(pbounds: Dict[str, Tuple[float, float]], params: Dict[str, float]) -> Dict[str, float]:
    """Clip parameter values to pbounds to satisfy optimizer.register constraints."""
    clipped: Dict[str, float] = {}
    for key, value in params.items():
        if key in pbounds:
            lo, hi = pbounds[key]
            if value < lo:
                clipped[key] = lo
            elif value > hi:
                clipped[key] = hi
            else:
                clipped[key] = value
        else:
            clipped[key] = value
    return clipped

def _load_bm25_seed_results(path: Path) -> List[Tuple[Dict[str, float], float]]:
    """Load BM25-only BO results to seed current optimizer.

    Returns a list of (params, target) tuples.
    """
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
        # Flexible extraction of results list
        if isinstance(data, dict):
            if 'all_results' in data:
                results = data['all_results']
            elif 'res' in data:
                results = data['res']
            elif 'results' in data:
                results = data['results']
            else:
                # Possibly a dict mapping or incompatible format
                return []
        elif isinstance(data, list):
            results = data
        else:
            return []

        seeds: List[Tuple[Dict[str, float], float]] = []
        for entry in results:
            params = entry.get('params', {}) if isinstance(entry, dict) else {}
            target = entry.get('target', None) if isinstance(entry, dict) else None
            if not params:
                continue
            # Map possible legacy keys to current names
            mapped = {
                'chunk_size_bm25': params.get('chunk_size_bm25', params.get('chunk_size', params.get('bm25_chunk_size'))),
                'overlap_bm25': params.get('overlap_bm25', params.get('overlap', params.get('bm25_overlap'))),
                'k1': params.get('k1'),
                'b': params.get('b'),
            }
            # Ensure required values present
            if any(v is None for v in mapped.values()):
                continue
            seeds.append((mapped, float(target) if target is not None else None))
        return seeds
    except Exception:
        return []

def _write_results(optimizer: BayesianOptimization, pbounds: Dict[str, Tuple[float, float]]) -> None:
    """Persist current optimization state to JSON and append latest row to CSV."""
    if not optimizer.res:
        return
    # JSON snapshot
    results = {
        'best_config': optimizer.max,
        'all_results': optimizer.res,
        'bounds': pbounds
    }
    RESULTS_SNAPSHOT_FILE.write_text(json.dumps(results, indent=2, default=str))

    # CSV append (append latest observation)
    latest = optimizer.res[-1]
    params: Dict[str, float] = latest.get('params', {})
    target = latest.get('target', None)
    # Cast to the actual values used during evaluation
    def _safe_get(key: str, default: float = 0.0) -> float:
        v = params.get(key)
        return float(v) if v is not None else float(default)

    cs_bm25 = int(round(_safe_get('chunk_size_bm25')))
    ov_bm25 = int(round(_safe_get('overlap_bm25')))
    cs_embed = int(round(_safe_get('chunk_size_embed')))
    ov_embed = int(round(_safe_get('overlap_embed')))
    # Clamp overlaps to be < chunk size
    cs_bm25 = max(2, cs_bm25)
    cs_embed = max(2, cs_embed)
    ov_bm25 = max(0, min(ov_bm25, cs_bm25 - 1))
    ov_embed = max(0, min(ov_embed, cs_embed - 1))
    model_sel_raw = _safe_get('model_selector')
    model_selector_cast = 0.0 if model_sel_raw < 0.5 else 1.0
    columns = [
        'target',
        'chunk_size_bm25', 'overlap_bm25', 'k1', 'b',
        'chunk_size_embed', 'overlap_embed', 'alpha', 'beta', 'model_selector',
    ]
    row = {
        'target': target,
        'chunk_size_bm25': cs_bm25,
        'overlap_bm25': ov_bm25,
        'k1': _safe_get('k1'),
        'b': _safe_get('b'),
        'chunk_size_embed': cs_embed,
        'overlap_embed': ov_embed,
        'alpha': _safe_get('alpha'),
        'beta': _safe_get('beta'),
        'model_selector': model_selector_cast,
    }
    header_needed = not RESULTS_CSV_FILE.exists()
    with RESULTS_CSV_FILE.open('a', encoding='utf-8') as f:
        if header_needed:
            f.write(','.join(columns) + '\n')
        values = [row.get(c, "") for c in columns]
        values_str = ["" if v is None else str(v) for v in values]
        f.write(','.join(values_str) + '\n')

def _append_csv_row_from_params(params: Dict[str, float], target: Optional[float]) -> None:
    """Append a single CSV row using cast values, without touching optimizer state."""
    def _safe_get(key: str, default: float = 0.0) -> float:
        v = params.get(key)
        return float(v) if v is not None else float(default)

    cs_bm25 = int(round(_safe_get('chunk_size_bm25')))
    ov_bm25 = int(round(_safe_get('overlap_bm25')))
    cs_embed = int(round(_safe_get('chunk_size_embed')))
    ov_embed = int(round(_safe_get('overlap_embed')))
    cs_bm25 = max(2, cs_bm25)
    cs_embed = max(2, cs_embed)
    ov_bm25 = max(0, min(ov_bm25, cs_bm25 - 1))
    ov_embed = max(0, min(ov_embed, cs_embed - 1))
    model_sel_raw = _safe_get('model_selector')
    model_selector_cast = 0.0 if model_sel_raw < 0.5 else 1.0

    columns = [
        'target',
        'chunk_size_bm25', 'overlap_bm25', 'k1', 'b',
        'chunk_size_embed', 'overlap_embed', 'alpha', 'beta', 'model_selector',
    ]
    row = {
        'target': target,
        'chunk_size_bm25': cs_bm25,
        'overlap_bm25': ov_bm25,
        'k1': _safe_get('k1'),
        'b': _safe_get('b'),
        'chunk_size_embed': cs_embed,
        'overlap_embed': ov_embed,
        'alpha': _safe_get('alpha'),
        'beta': _safe_get('beta'),
        'model_selector': model_selector_cast,
    }
    header_needed = not RESULTS_CSV_FILE.exists()
    with RESULTS_CSV_FILE.open('a', encoding='utf-8') as f:
        if header_needed:
            f.write(','.join(columns) + '\n')
        values = [row.get(c, "") for c in columns]
        values_str = ["" if v is None else str(v) for v in values]
        f.write(','.join(values_str) + '\n')

def run_bayesian_optimization(
    init_points: int = 50,
    n_iter: int = 1000,
    use_bm25_seeds: bool = True,
    bm25_seed_file: Path = Path(".cache/bo_results.json"),
    fixed_bm25: Optional[Dict[str, float]] = None,
):
    """Run Bayesian optimization for hybrid retrieval."""
    # Define search space
    pbounds = {
        "chunk_size_bm25": (75, 300),
        "overlap_bm25": (1, 50),
        "k1": (0.0, 3.0),
        "b": (0.1, 1.5),
        "chunk_size_embed": (75, 300),
        "overlap_embed": (1, 50),
        "alpha": (0.0, 100.0),
        "beta": (0.0, 100.0),
        "model_selector": (0, 1),  # Discrete toggle between MiniLM (0) and ColBERT (1)
    }

    # Wrap objective if BM25 is fixed
    wrapped_f = objective_function
    if fixed_bm25 is not None:
        fx_cs = int(round(float(fixed_bm25.get('chunk_size') or fixed_bm25.get('chunk_size_bm25'))))
        fx_ov = int(round(float(fixed_bm25.get('overlap') or fixed_bm25.get('overlap_bm25'))))
        fx_k1 = float(fixed_bm25.get('k1'))
        fx_b = float(fixed_bm25.get('b'))
        # Clamp bounds to those fixed values
        pbounds['chunk_size_bm25'] = (fx_cs, fx_cs)
        pbounds['overlap_bm25'] = (fx_ov, fx_ov)
        pbounds['k1'] = (fx_k1, fx_k1)
        pbounds['b'] = (fx_b, fx_b)

        def wrapped_f(
            chunk_size_bm25,
            overlap_bm25,
            k1,
            b,
            chunk_size_embed,
            overlap_embed,
            alpha,
            beta,
            model_selector,
        ):
            return objective_function(
                fx_cs, fx_ov, fx_k1, fx_b,
                chunk_size_embed, overlap_embed,
                alpha, beta, model_selector
            )

    # Initialize optimizer
    optimizer = BayesianOptimization(
        f=wrapped_f,
        pbounds=pbounds,
        random_state=42,
        verbose=1  # Reduced verbosity since we have our own progress tracking
    )

    # Make the GP very smooth (super long kernel length) to bias exploitation
    # Fix kernel hyperparameters to avoid re-fitting to shorter scales
    long_kernel = C(1.0, constant_value_bounds='fixed') * RBF(length_scale=1e5, length_scale_bounds='fixed')
    try:
        optimizer.set_gp_params(kernel=long_kernel, alpha=1e-6, normalize_y=True)
    except Exception:
        # Fallback in case set_gp_params signature differs
        pass

    # Optionally register BM25-only seed observations
    num_seeds_registered = 0
    if use_bm25_seeds:
        seeds = _load_bm25_seed_results(bm25_seed_file)
        for legacy_params, target in seeds:
            if target is None:
                continue
            # Expand to full param dict for current space
            full_params = {
                'chunk_size_bm25': int(round(legacy_params['chunk_size_bm25'])),
                'overlap_bm25': int(round(legacy_params['overlap_bm25'])),
                'k1': float(legacy_params['k1']),
                'b': float(legacy_params['b']),
                # Unused when beta≈0, but must be present and within bounds
                'chunk_size_embed': 128.0,
                'overlap_embed': 32.0,
                'alpha': 1.0,
                'beta': 0.0,
                'model_selector': 0.0,
            }
            full_params = _clip_to_bounds(pbounds, full_params)
            try:
                optimizer.register(params=full_params, target=float(target))
                num_seeds_registered += 1
                # Immediately persist seed observation to CSV and snapshot
                _append_csv_row_from_params(full_params, float(target))
            except Exception:
                # If any seed is out-of-bounds or duplicate, skip it
                continue
        # Write a snapshot JSON including the seeded observations
        RESULTS_SNAPSHOT_FILE.write_text(json.dumps({
            'best_config': optimizer.max,
            'all_results': optimizer.res,
            'bounds': pbounds
        }, indent=2, default=str))

    # Run optimization with progress tracking
    total_trials = init_points + n_iter
    print(f"🚀 Starting Bayesian optimization...")
    print(f"   Initial random points: {init_points}")
    print(f"   Optimization iterations: {n_iter}")
    print(f"   Total trials: {total_trials}")
    print(f"   Search space: {len(pbounds)} parameters")
    if use_bm25_seeds:
        print(f"   Seeded BM25-only observations: {num_seeds_registered}")
    print("="*60)
    
    # Run optimization with per-step persistence
    print("Starting optimization...")
    # Phase 1: random exploration only (save after each random trial)
    if init_points > 0:
        with tqdm(total=init_points, desc="Random init", unit="trial") as pbar_init:
            for _ in range(init_points):
                optimizer.maximize(init_points=1, n_iter=0)
                _write_results(optimizer, pbounds)
                pbar_init.update(1)

    # Phase 2: guided iterations with dual progress bars
    with tqdm(total=n_iter, desc="BO total", unit="iter") as pbar_total:
        for i_iter in range(n_iter):
            # inner bar for evaluation progress (statements)
            inner_bar: Optional[tqdm] = None
            def _on_eval_progress(done: int) -> None:
                if inner_bar is not None:
                    inner_bar.n = done
                    inner_bar.refresh()

            # bind callback
            global EVAL_PROGRESS_CALLBACK
            EVAL_PROGRESS_CALLBACK = None  # disabled by default; we create it lazily below

            # Run a single BO step and show inner evaluation progress bar if possible
            # We cannot easily hook into optimizer's internal call, so we just run one step
            # and show a separate inner bar during evaluate_config via the global callback.
            # Create the bar at the expected size (len(statements)) only when evaluate_config starts.
            # Since we cannot know 'total' ahead easily, we initialize on first callback.
            try:
                # Create a proxy to lazily create inner tqdm when first called
                created = {'bar': False}
                def cb(done: int):
                    nonlocal inner_bar
                    if not created['bar']:
                        # Default to 200 if unknown; bar will adjust via .total if needed
                        inner_bar = tqdm(total=200, desc="Eval run", unit="query", leave=False)
                        created['bar'] = True
                    _on_eval_progress(done)
                EVAL_PROGRESS_CALLBACK = cb
                optimizer.maximize(init_points=0, n_iter=1)
            finally:
                # cleanup inner bar and callback
                EVAL_PROGRESS_CALLBACK = None
                if inner_bar is not None:
                    inner_bar.close()
            _write_results(optimizer, pbounds)
            pbar_total.update(1)
    
    # Print progress summary
    print(f"\nOptimization complete! Best accuracy: {optimizer.max['target']:.4f}")

    # Save results
    results = {
        'best_config': optimizer.max,
        'all_results': optimizer.res,
        'bounds': pbounds
    }
    
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("\nBest Configuration Found:")
    print(f"Top-1 Accuracy: {optimizer.max['target']:.4f}")
    print("Parameters:")
    for param, value in optimizer.max['params'].items():
        print(f"  {param}: {value}")

    return optimizer

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Run Bayesian optimization
    optimizer = run_bayesian_optimization(init_points=0, n_iter=1000)
    
    # Print summary
    print("\n" + "="*50)
    print("HYBRID RETRIEVAL OPTIMIZATION COMPLETE")
    print("="*50)
    print(f"Results saved to: {RESULTS_FILE}")