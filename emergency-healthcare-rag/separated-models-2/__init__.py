#!/usr/bin/env python3
"""
Separated Models 2 - BM25-Only RAG System
Emergency Healthcare RAG with optimized BM25 search
"""

from .model import create_rag_model, EmergencyHealthcareRAG
from .search import bm25_search, get_best_topic, get_rich_context_for_statement
from .llm import create_llm_interface, LLMInterface
from .config import get_llm_model, list_available_models, set_llm_model

__version__ = "2.0.0"
__author__ = "Emergency Healthcare RAG Team"

__all__ = [
    'create_rag_model',
    'EmergencyHealthcareRAG',
    'bm25_search',
    'get_best_topic',
    'get_rich_context_for_statement',
    'create_llm_interface',
    'LLMInterface',
    'get_llm_model',
    'list_available_models',
    'set_llm_model'
] 