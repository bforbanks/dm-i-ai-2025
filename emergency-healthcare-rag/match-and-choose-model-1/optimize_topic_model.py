#!/usr/bin/env python3
"""
Hierarchical Topic Model Optimizer (downloads models at start; offline thereafter)

What it does
- Immediately downloads the embedding models to a local folder at startup (or reuses cached copies).
- Evaluates BM25, Semantic (SentenceTransformers), and Hybrid (linear, RRF) with topic-level aggregation (max over chunks).
- Metrics: top1, top2, top3, avg_rank, mean margin@1 (gap between top-1 and top-2). Misses are recorded as rank=11.
- Writes ONLY two small files, updated after every config:
    models_90_plus.json  (all configs with acc >= 0.90; metrics + config)
    top_5_models.json    (best five so far; metrics + config)
- Optional randomized search order and minutes budget. Safe to stop at any time; files reflect best-so-far.

Run example
python match-and-choose-model-1/optimize_topic_model.py \
  --data-dir /path/to/data \
  --models-dir /path/to/local_models \
  --random-search --max-samples 50 --minutes-budget 60
"""

import os, json, time, warnings, argparse, random
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning, module="torch.*")
warnings.filterwarnings("ignore", message=".*encoder_attention_mask.*")

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch

try:
    from tqdm import tqdm
    tqdm.monitor_interval = 0
    USE_TQDM = True
except Exception:
    USE_TQDM = False
    def tqdm(it, desc="", leave=True, **kw):
        total = len(it) if hasattr(it, '__len__') else None
        for i, x in enumerate(it):
            if total and i % max(1, total // 20) == 0:
                print(f"{desc}: {i+1}/{total} ({100*(i+1)/total:.0f}%)")
            yield x

# ---------------------------- Paths ----------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR: Path                 # set in __main__
MODELS_DIR: Path               # set in __main__
OUT_90P = BASE_DIR / "models_90_plus.json"
OUT_TOP5 = BASE_DIR / "top_5_models.json"
CACHE_DIR = BASE_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)

# ---------------------------- Config ----------------------------
@dataclass
class OptimizationConfig:
    bm25_chunk_sizes: List[int] = None
    bm25_overlap_ratios: List[float] = None
    embedding_models: List[str] = None
    embedding_chunk_sizes: List[int] = None
    embedding_overlap_ratios: List[float] = None
    fusion_strategies: List[str] = None
    top_bm25_configs: int = 5
    top_embedding_configs: int = 5
    zoom_enabled: bool = False
    zoom_radius: int = 2
    use_condensed_topics: bool = True
    max_samples: int = 50
    cache_embeddings: bool = True
    save_detailed_results: bool = False
    display_every: int = 20
    random_search: bool = False
    max_configs_phase1: Optional[int] = None
    max_configs_phase2: Optional[int] = None
    max_configs_phase3: Optional[int] = None
    seed: int = 42
    def __post_init__(self):
        self.bm25_chunk_sizes = self.bm25_chunk_sizes or [96,112,128,144,160,176]
        self.bm25_overlap_ratios = self.bm25_overlap_ratios or [0.0]
        self.embedding_models = self.embedding_models or [
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/all-mpnet-base-v2",
        ]
        self.embedding_chunk_sizes = self.embedding_chunk_sizes or [96,112,128,144,160]
        self.embedding_overlap_ratios = self.embedding_overlap_ratios or [0.0]
        self.fusion_strategies = self.fusion_strategies or ["bm25_only","semantic_only","linear_0.3","linear_0.5","linear_0.7","rrf"]

config = OptimizationConfig()

# ---------------------------- IO helpers ----------------------------
def _read_json(p: Path, default):
    try: return json.loads(p.read_text())
    except Exception: return default

def _write_json(p: Path, obj):
    p.write_text(json.dumps(obj, indent=2))

def update_high_performance_models(result: Dict):
    """Append any >=0.90 accuracy config (metrics + config) and highlight in terminal."""
    acc = result.get('metrics', {}).get('accuracy', 0.0)
    if acc < 0.90: return
    entries = _read_json(OUT_90P, [])
    m, c = result['metrics'], result['config']
    entry = {
        "timestamp": int(time.time()),
        "accuracy": m['accuracy'],
        "top2_accuracy": m['top2_accuracy'],
        "top3_accuracy": m['top3_accuracy'],
        "avg_rank": m['avg_rank'],
        "mean_margin_at_1": m.get('mean_margin_at_1'),
        "config": c
    }
    key = (json.dumps(c, sort_keys=True), round(m['accuracy'], 6))
    if not any((json.dumps(e["config"], sort_keys=True), round(e["accuracy"],6)) == key for e in entries):
        entries.append(entry)
        _write_json(OUT_90P, entries)
    print("\033[92m" + "🎯  ACCURACY ≥ 0.90 — saved to models_90_plus.json" + "\033[0m")

def update_top_5_models(all_results: List[Dict]):
    """Rewrite current top five"""
    if not all_results: return None
    s = sorted(all_results, key=lambda x: x['metrics']['accuracy'], reverse=True)[:5]
    out = {"last_updated": int(time.time()),
           "total_configurations_tested": len(all_results),
           "top_5_models": []}
    for i, r in enumerate(s, 1):
        m, c = r['metrics'], r['config']
        out["top_5_models"].append({
            "rank": i,
            "accuracy": m['accuracy'],
            "top2_accuracy": m['top2_accuracy'],
            "top3_accuracy": m['top3_accuracy'],
            "avg_rank": m['avg_rank'],
            "mean_margin_at_1": m.get('mean_margin_at_1'),
            "config": c
        })
    _write_json(OUT_TOP5, out)
    return out

def display_top_5_models(all_results: List[Dict], i: int, total: int, force=False):
    if not force and (i % config.display_every != 0): return
    top = update_top_5_models(all_results)
    if not top:
        print("No results yet; top-5 unavailable.")
        return
    print(f"\nTOP 5 at {time.strftime('%H:%M:%S')} [{i}/{total} = {100*i/total:.1f}%]")
    for t in top["top_5_models"]:
        print(f"{t['rank']}. acc={t['accuracy']:.3f} top2={t['top2_accuracy']:.3f} top3={t['top3_accuracy']:.3f} avg_rank={t['avg_rank']:.2f} margin@1={t['mean_margin_at_1']}")
    print(f"Saved to {OUT_TOP5.name}")

# ---------------------------- Data & metrics ----------------------------
def _resolve_data_dir(cli_path: Optional[str]) -> Path:
    if cli_path: return Path(cli_path).resolve()
    env = os.environ.get("EHR_DATA_DIR")
    if env: return Path(env).resolve()
    cands = [BASE_DIR/"data", BASE_DIR.parent/"data", Path.cwd()/"data"]
    for c in cands:
        if (c/"topics.json").exists(): return c.resolve()
    raise FileNotFoundError("Provide --data-dir or set EHR_DATA_DIR to folder containing topics.json")

def _require_data_layout():
    need = [DATA_DIR / "topics.json", DATA_DIR / "train" / "statements", DATA_DIR / "train" / "answers"]
    miss = [str(p) for p in need if not p.exists()]
    if miss:
        raise FileNotFoundError("Required data paths missing:\n  " + "\n  ".join(miss))

def chunk_words(words: List[str], size: int, overlap_tokens: int) -> List[str]:
    if len(words) <= size: return [' '.join(words)]
    step = max(1, size - overlap_tokens)
    out = [' '.join(words[i:i+size]) for i in range(0, len(words) - size + 1, step)]
    return out or [' '.join(words[:size])]

def load_statements() -> List[Tuple[str, int]]:
    sdir = DATA_DIR / "train" / "statements"
    adir = DATA_DIR / "train" / "answers"
    recs = []
    for p in sorted(sdir.glob("*.txt")):
        sid = p.stem.split("_")[1]
        recs.append((p.read_text().strip(), json.loads((adir / f"statement_{sid}.json").read_text())["statement_topic"]))
    return recs[:config.max_samples]

def _aggregate_topic_scores(scores: np.ndarray, doc_ids: List[int]) -> Dict[int, float]:
    agg: Dict[int, float] = {}
    for s, tid in zip(scores, doc_ids):
        if tid not in agg or s > agg[tid]: agg[tid] = float(s)
    return agg

def _rank_and_margin(topic_scores: Dict[int, float], true_topic: int) -> Tuple[int, float]:
    order = [t for t, _ in sorted(topic_scores.items(), key=lambda kv: kv[1], reverse=True)][:10]
    rank = 11
    for i, t in enumerate(order, 1):
        if t == true_topic: rank = i; break
    s1 = topic_scores[order[0]] if order else 0.0
    s2 = topic_scores[order[1]] if len(order) > 1 else 0.0
    return rank, float(s1 - s2)

def calculate_metrics(results: List[Dict]) -> Dict:
    if not results:
        return {"accuracy":0.0,"top2_accuracy":0.0,"top3_accuracy":0.0,"avg_rank":0.0,"total_samples":0}
    total = len(results)
    top1 = sum(1 for r in results if r['rank_correct']==1)
    top2 = sum(1 for r in results if 1 <= r['rank_correct'] <= 2)
    top3 = sum(1 for r in results if 1 <= r['rank_correct'] <= 3)
    avg_rank = sum(r['rank_correct'] for r in results)/total
    mean_margin = float(np.mean([r['margin_at_1'] for r in results]))
    return {"accuracy":top1/total,"top2_accuracy":top2/total,"top3_accuracy":top3/total,
            "avg_rank":avg_rank,"total_samples":total,"mean_margin_at_1":mean_margin}

def _norm(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    mn, mx = float(v.min()), float(v.max())
    return (v - mn)/(mx - mn) if mx - mn > 1e-8 else np.zeros_like(v)

# ---------------------------- Models: download at start; load from cache thereafter ----------------------------
def download_models_at_start(model_names: List[str]):
    """Download models to MODELS_DIR if missing; reuse cache if present.
       After this, we always load by repo id with cache_folder=MODELS_DIR (offline-safe)."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for name in model_names:
        try:
            # Attempt to instantiate with cache_folder; if cached it won't hit the network.
            _ = SentenceTransformer(name, cache_folder=str(MODELS_DIR))
            print(f"✔ Model available: {name}")
        except Exception as e:
            # Retry once verbosely
            print(f"Attempting download for {name} → {MODELS_DIR}")
            _ = SentenceTransformer(name, cache_folder=str(MODELS_DIR))
            print(f"✔ Downloaded: {name}")

def _cache_path(model: str, chunk: int, ov: float, cond: bool) -> Path:
    key = f"{model.replace('/','_')}__c{chunk}__o{ov:.3f}__cond{int(cond)}.npz"
    return CACHE_DIR / key

# ---------------------------- Phase 1: BM25 ----------------------------
def build_bm25_index(chunk_size: int, overlap_ratio: float, use_condensed_topics: bool=True) -> Dict:
    tdir = DATA_DIR / ("condensed_topics" if use_condensed_topics else "topics")
    tmap = json.loads((DATA_DIR / "topics.json").read_text())
    docs, doc_ids = [], []
    for tname, tid in tmap.items():
        if use_condensed_topics:
            d = tdir / tname
            if d.is_dir():
                for md in d.glob("*.md"):
                    words = md.read_text().split()
                    chunks = chunk_words(words, chunk_size, int(chunk_size*overlap_ratio))
                    docs.extend(chunks); doc_ids.extend([tid]*len(chunks))
        else:
            p = tdir / f"{tid}.md"
            if p.exists():
                words = p.read_text().split()
                chunks = chunk_words(words, chunk_size, int(chunk_size*overlap_ratio))
                docs.extend(chunks); doc_ids.extend([tid]*len(chunks))
    if not docs: raise ValueError("No documents for BM25")
    tokenized = [d.split() for d in docs]
    return {"bm25": BM25Okapi(tokenized), "doc_ids": doc_ids,
            "chunk_size": chunk_size, "overlap": overlap_ratio}

def evaluate_bm25_config(bm25_data: Dict, statements: List[Tuple[str,int]]) -> Dict:
    bm25, doc_ids = bm25_data["bm25"], bm25_data["doc_ids"]
    results = []
    for s, true_t in tqdm(statements, desc="BM25", disable=not USE_TQDM):
        scores = _norm(bm25.get_scores(s.split()))
        topic_scores = _aggregate_topic_scores(scores, doc_ids)
        rank, margin = _rank_and_margin(topic_scores, true_t)
        results.append({"rank_correct": rank, "margin_at_1": margin})
    metrics = calculate_metrics(results)
    return {"config":{"chunk_size":bm25_data["chunk_size"],"overlap":bm25_data["overlap"],"strategy":"bm25_only"},
            "metrics":metrics, "results": results if config.save_detailed_results else []}

def run_bm25_optimization(statements: List[Tuple[str,int]], deadline: Optional[float]) -> List[Dict]:
    grid = [(cs, ov) for cs in config.bm25_chunk_sizes for ov in config.bm25_overlap_ratios]
    if config.random_search:
        random.Random(config.seed).shuffle(grid)
    if config.max_configs_phase1:
        grid = grid[:config.max_configs_phase1]

    all_results, total = [], len(grid)
    print("PHASE 1: BM25")
    for k, (cs, ov) in enumerate(grid, 1):
        print(f"\nBM25 {k}/{total}: chunk={cs} overlap_ratio={ov:.2f}")
        try:
            data = build_bm25_index(cs, ov, config.use_condensed_topics)
            res = evaluate_bm25_config(data, statements)
            all_results.append(res)
            m = res['metrics']
            print(f"✅ acc={m['accuracy']:.3f} top2={m['top2_accuracy']:.3f} top3={m['top3_accuracy']:.3f} avg_rank={m['avg_rank']:.2f}")
            if m['accuracy'] >= 0.90: update_high_performance_models(res)
            update_top_5_models(all_results); display_top_5_models(all_results, k, total)
        except Exception as e:
            print(f"Error: {e}")
        if deadline and time.time() > deadline:
            print("Time budget reached during Phase 1.")
            break
    all_results.sort(key=lambda x: x['metrics']['accuracy'], reverse=True)
    display_top_5_models(all_results, min(len(grid), k), total, force=True)
    return all_results[:config.top_bm25_configs] if all_results else []

# ---------------------------- Phase 2: Embeddings ----------------------------
def build_semantic_index(model_name: str, chunk_size: int, overlap_ratio: float, use_condensed_topics: bool=True) -> Dict:
    cp = _cache_path(model_name, chunk_size, overlap_ratio, use_condensed_topics)
    if config.cache_embeddings and cp.exists():
        nz = np.load(cp, allow_pickle=True)
        return {"model": None, "embeddings": nz["embeddings"], "doc_ids": nz["doc_ids"].tolist(),
                "chunk_size": chunk_size, "overlap": overlap_ratio, "model_name": model_name}

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Load by repo id from our MODELS_DIR cache (offline-safe if already downloaded)
    model = SentenceTransformer(model_name, device=device, cache_folder=str(MODELS_DIR))

    tdir = DATA_DIR / ("condensed_topics" if use_condensed_topics else "topics")
    tmap = json.loads((DATA_DIR / "topics.json").read_text())
    docs, doc_ids = [], []
    for tname, tid in tmap.items():
        if use_condensed_topics:
            d = tdir / tname
            if d.is_dir():
                for md in d.glob("*.md"):
                    words = md.read_text().split()
                    chunks = chunk_words(words, chunk_size, int(chunk_size*overlap_ratio))
                    docs.extend(chunks); doc_ids.extend([tid]*len(chunks))
        else:
            p = tdir / f"{tid}.md"
            if p.exists():
                words = p.read_text().split()
                chunks = chunk_words(words, chunk_size, int(chunk_size*overlap_ratio))
                docs.extend(chunks); doc_ids.extend([tid]*len(chunks))

    if not docs: raise ValueError("No documents for embeddings")
    print(f"Embedding {len(docs)} chunks with {model_name.split('/')[-1]} on {device}...")
    emb = model.encode(docs, convert_to_numpy=True, show_progress_bar=USE_TQDM)
    if config.cache_embeddings:
        np.savez_compressed(cp, embeddings=emb, doc_ids=np.array(doc_ids))
    return {"model": model, "embeddings": emb, "doc_ids": doc_ids,
            "chunk_size": chunk_size, "overlap": overlap_ratio, "model_name": model_name}

def evaluate_semantic_config(sd: Dict, statements: List[Tuple[str,int]]) -> Dict:
    if sd["model"] is None:
        model = SentenceTransformer(sd["model_name"], device='cpu', cache_folder=str(MODELS_DIR))
    else:
        model = sd["model"]
    emb, doc_ids = sd["embeddings"], sd["doc_ids"]
    results = []
    for s, true_t in tqdm(statements, desc="Semantic", disable=not USE_TQDM):
        q = model.encode([s], convert_to_numpy=True)
        sims = _norm(cosine_similarity(q, emb)[0])
        topic_scores = _aggregate_topic_scores(sims, doc_ids)
        rank, margin = _rank_and_margin(topic_scores, true_t)
        results.append({"rank_correct": rank, "margin_at_1": margin})
    metrics = calculate_metrics(results)
    return {"config":{"model_name":sd["model_name"],"chunk_size":sd["chunk_size"],"overlap":sd["overlap"],"strategy":"semantic_only"},
            "metrics":metrics, "results": results if config.save_detailed_results else []}

def run_embedding_optimization(statements: List[Tuple[str,int]], deadline: Optional[float]) -> List[Dict]:
    grid = [(mn, cs, ov) for mn in config.embedding_models
                         for cs in config.embedding_chunk_sizes
                         for ov in config.embedding_overlap_ratios]
    if config.random_search:
        random.Random(config.seed).shuffle(grid)
    if config.max_configs_phase2:
        grid = grid[:config.max_configs_phase2]

    all_results, total = [], len(grid)
    print("\nPHASE 2: Embeddings")
    for k, (mn, cs, ov) in enumerate(grid, 1):
        print(f"\nEMB {k}/{total}: {mn.split('/')[-1]} chunk={cs} overlap_ratio={ov:.2f}")
        try:
            sd = build_semantic_index(mn, cs, ov, config.use_condensed_topics)
            res = evaluate_semantic_config(sd, statements)
            all_results.append(res)
            m = res['metrics']
            print(f"✅ acc={m['accuracy']:.3f} top2={m['top2_accuracy']:.3f} top3={m['top3_accuracy']:.3f} avg_rank={m['avg_rank']:.2f}")
            if m['accuracy'] >= 0.90: update_high_performance_models(res)
            update_top_5_models(all_results); display_top_5_models(all_results, k, total)
        except Exception as e:
            print(f"Error: {e}")
        if deadline and time.time() > deadline:
            print("Time budget reached during Phase 2.")
            break
    all_results.sort(key=lambda x: x['metrics']['accuracy'], reverse=True)
    display_top_5_models(all_results, min(len(grid), k), total, force=True)
    return all_results[:config.top_embedding_configs] if all_results else []

# ---------------------------- Phase 3: Hybrid ----------------------------
class HybridSearcher:
    def __init__(self, bm25_data: Dict, sem_data: Dict):
        self.bm25 = bm25_data["bm25"]
        self.doc_ids = bm25_data["doc_ids"]
        if sem_data["model"] is None:
            self.model = SentenceTransformer(sem_data["model_name"], device='cpu', cache_folder=str(MODELS_DIR))
        else:
            self.model = sem_data["model"]
        self.emb = sem_data["embeddings"]
    def linear(self, q: str, w: float=0.5) -> Dict[int,float]:
        b = _norm(self.bm25.get_scores(q.split()))
        s = _norm(cosine_similarity(self.model.encode([q], convert_to_numpy=True), self.emb)[0])
        return _aggregate_topic_scores(w*b + (1-w)*s, self.doc_ids)
    def rrf(self, q: str, k: int=60) -> Dict[int,float]:
        b = _norm(self.bm25.get_scores(q.split())); ib = np.argsort(b)[::-1]
        s = _norm(cosine_similarity(self.model.encode([q], convert_to_numpy=True), self.emb)[0]); is_ = np.argsort(s)[::-1]
        out: Dict[int,float] = {}
        for r,i in enumerate(ib): out[self.doc_ids[i]] = out.get(self.doc_ids[i],0.0) + 1.0/(k+r+1)
        for r,i in enumerate(is_): out[self.doc_ids[i]] = out.get(self.doc_ids[i],0.0) + 1.0/(k+r+1)
        return out

def evaluate_hybrid_config(bc: Dict, sc: Dict, strat: str, statements: List[Tuple[str,int]]) -> Dict:
    bd = build_bm25_index(bc['chunk_size'], bc['overlap'], config.use_condensed_topics)
    sd = build_semantic_index(sc['model_name'], sc['chunk_size'], sc['overlap'], config.use_condensed_topics)
    hs = HybridSearcher(bd, sd)
    results = []
    for s, true_t in tqdm(statements, desc="Hybrid", disable=not USE_TQDM):
        if strat == "bm25_only":
            topic_scores = _aggregate_topic_scores(_norm(bd["bm25"].get_scores(s.split())), bd["doc_ids"])
        elif strat == "semantic_only":
            sims = _norm(cosine_similarity(hs.model.encode([s], convert_to_numpy=True), sd["embeddings"])[0])
            topic_scores = _aggregate_topic_scores(sims, sd["doc_ids"])
        elif strat.startswith("linear_"):
            topic_scores = hs.linear(s, float(strat.split("_")[1]))
        elif strat == "rrf":
            topic_scores = hs.rrf(s)
        else:
            continue
        rank, margin = _rank_and_margin(topic_scores, true_t)
        results.append({"rank_correct": rank, "margin_at_1": margin})
    metrics = calculate_metrics(results)
    return {"config":{"bm25_config":bc,"semantic_config":sc,"fusion_strategy":strat},
            "metrics":metrics, "results": results if config.save_detailed_results else []}

def run_hybrid_optimization(bm25_results: List[Dict], sem_results: List[Dict],
                            statements: List[Tuple[str,int]], deadline: Optional[float]) -> List[Dict]:
    if not bm25_results or not sem_results: return []
    combos = [(b['config'], s['config'], strat)
              for b in bm25_results for s in sem_results for strat in config.fusion_strategies]
    if config.random_search:
        random.Random(config.seed).shuffle(combos)
    if config.max_configs_phase3:
        combos = combos[:config.max_configs_phase3]

    all_results, total = [], len(combos)
    print("\nPHASE 3: Hybrid")
    for k, (bc, sc, strat) in enumerate(combos, 1):
        print(f"\nHYB {k}/{total}: {strat} | BM25 chunk={bc['chunk_size']} ov={bc['overlap']:.2f} | SEM {sc['model_name'].split('/')[-1]} chunk={sc['chunk_size']} ov={sc['overlap']:.2f}")
        try:
            res = evaluate_hybrid_config(bc, sc, strat, statements)
            all_results.append(res)
            m = res['metrics']
            print(f"✅ acc={m['accuracy']:.3f} top2={m['top2_accuracy']:.3f} top3={m['top3_accuracy']:.3f} avg_rank={m['avg_rank']:.2f}")
            if m['accuracy'] >= 0.90: update_high_performance_models(res)
            update_top_5_models(all_results); display_top_5_models(all_results, k, total)
        except Exception as e:
            print(f"Error: {e}")
        if deadline and time.time() > deadline:
            print("Time budget reached during Phase 3.")
            break
    all_results.sort(key=lambda x: x['metrics']['accuracy'], reverse=True)
    display_top_5_models(all_results, min(len(combos), k), total, force=True)
    return all_results

# ---------------------------- Phase 4: Zoom (optional, off by default) ----------------------------
def generate_zoom_configs(prom: Dict, radius: int=2) -> List[Dict]:
    base = prom['config']; bc, bo = base['bm25_config']['chunk_size'], base['bm25_config']['overlap']
    out = []
    for dc in range(-radius, radius+1):
        for do in range(-radius, radius+1):
            nc, no = bc + dc*8, round(bo + do*0.05, 3)
            if 64 <= nc <= 256 and 0.0 <= no <= 0.4:
                out.append({"bm25_config":{**base['bm25_config'],"chunk_size":nc,"overlap":no},
                            "semantic_config":base['semantic_config'],
                            "fusion_strategy":base['fusion_strategy']})
    return out

def run_zoom_optimization(promising: List[Dict], statements: List[Tuple[str,int]], deadline: Optional[float]) -> List[Dict]:
    if not config.zoom_enabled or not promising: return promising or []
    print("\nPHASE 4: Zoom")
    zoom_results = []
    for i, pr in enumerate(promising[:3], 1):
        print(f"\nZoom {i}: {pr['config']['fusion_strategy']} acc={pr['metrics']['accuracy']:.3f}")
        zcfgs = generate_zoom_configs(pr, config.zoom_radius)
        print(f"variants={len(zcfgs)}")
        for j, zc in enumerate(zcfgs, 1):
            try:
                res = evaluate_hybrid_config(zc['bm25_config'], zc['semantic_config'], zc['fusion_strategy'], statements)
                zoom_results.append(res)
                if j % 10 == 0:
                    print(f"  {j}/{len(zcfgs)} acc={res['metrics']['accuracy']:.3f}")
                    update_top_5_models(zoom_results)
            except Exception as e:
                print(f"  Zoom error {j}: {e}")
            if deadline and time.time() > deadline:
                print("Time budget reached during Zoom.")
                break
        if deadline and time.time() > deadline:
            break
    all_results = (promising + zoom_results)
    all_results.sort(key=lambda x: x['metrics']['accuracy'], reverse=True)
    return all_results

# ---------------------------- Main ----------------------------
def run_optimization(minutes_budget: Optional[int]):
    start = time.time()
    deadline = start + 60*minutes_budget if minutes_budget else None

    _require_data_layout()
    statements = load_statements()

    print("Starting optimization")

    # Phase 1
    bm25_results = run_bm25_optimization(statements, deadline)

    # Phase 2
    if deadline and time.time() > deadline:
        print("Budget exhausted after Phase 1.")
        return bm25_results
    sem_results = run_embedding_optimization(statements, deadline)

    # Phase 3
    if deadline and time.time() > deadline:
        print("Budget exhausted after Phase 2.")
        return sem_results
    hybrid_results = run_hybrid_optimization(bm25_results, sem_results, statements, deadline)

    # Phase 4 (optional; disabled by default)
    if config.zoom_enabled and (not deadline or time.time() < deadline):
        final_results = run_zoom_optimization(hybrid_results, statements, deadline)
    else:
        final_results = hybrid_results

    elapsed = time.time() - start
    print(f"\nDone in {elapsed/3600:.2f} h")
    if final_results:
        best = final_results[0]
        m = best['metrics']
        print("\nBEST CONFIG:")
        print(f" strategy={best['config']['fusion_strategy'] if 'fusion_strategy' in best['config'] else best['config'].get('strategy','')}")
        print(f" acc={m['accuracy']:.3f} top2={m['top2_accuracy']:.3f} top3={m['top3_accuracy']:.3f} avg_rank={m['avg_rank']:.2f} margin@1={m.get('mean_margin_at_1')}")
        update_high_performance_models(best)
        update_top_5_models(final_results)
    else:
        print("No results generated.")
    return final_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=None, help="Directory containing topics.json and train/")
    parser.add_argument("--models-dir", type=str, default=None, help="Directory to store/load SentenceTransformer models (will download here)")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--display-every", type=int, default=None)
    parser.add_argument("--random-search", action="store_true")
    parser.add_argument("--max-configs-phase1", type=int, default=None)
    parser.add_argument("--max-configs-phase2", type=int, default=None)
    parser.add_argument("--max-configs-phase3", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--minutes-budget", type=int, default=None, help="Stop after this many minutes")
    args = parser.parse_args()

    # Apply CLI -> globals
    DATA_DIR = _resolve_data_dir(args.data_dir)
    MODELS_DIR = Path(args.models_dir).resolve() if args.models_dir else (BASE_DIR / "models")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Config overrides
    if args.max_samples is not None: config.max_samples = args.max_samples
    if args.display_every is not None: config.display_every = args.display_every
    config.random_search = bool(args.random_search)
    config.max_configs_phase1 = args.max_configs_phase1
    config.max_configs_phase2 = args.max_configs_phase2
    config.max_configs_phase3 = args.max_configs_phase3
    config.seed = args.seed
    random.seed(config.seed); np.random.seed(config.seed)

    # 1) Download models at start (or reuse local cache)
    print(f"Ensuring models in: {MODELS_DIR}")
    download_models_at_start(config.embedding_models)

    # 2) After download, force offline mode for the remainder of the run (will load from cache_folder)
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"

    # 3) Run optimization under optional budget
    run_optimization(args.minutes_budget)
