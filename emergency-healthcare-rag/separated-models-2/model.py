#!/usr/bin/env python3
"""
Main model interface for separated-models-2
Combines BM25 search with LLM for emergency healthcare RAG
"""

from typing import Dict, Any, Optional
from search import get_best_topic, get_rich_context_for_statement
from llm import create_llm_interface

class EmergencyHealthcareRAG:
    def __init__(self, model_name: Optional[str] = None):
        self.llm = create_llm_interface(model_name)
        
    def process_statement(self, statement: str) -> Dict[str, Any]:
        """
        Process a medical statement through the RAG pipeline
        """
        # Step 1: Find the best topic using BM25 search
        topic_id = get_best_topic(statement)
        
        # Step 2: Get rich context for the statement
        context = get_rich_context_for_statement(statement, topic_id)
        
        # Step 3: Generate response using LLM
        prompt = self._build_prompt(statement, context)
        response = self.llm.generate_response(prompt)
        
        return {
            'statement': statement,
            'topic_id': topic_id,
            'context': context,
            'response': response,
            'model_used': self.llm.get_model_name()
        }
    
    def _build_prompt(self, statement: str, context: str) -> str:
        """
        Build the prompt for the LLM - returns only a number (1 for TRUE, 0 for FALSE)
        """
        return f"""You are a medical expert. Determine if the following statement is TRUE or FALSE based on the provided medical context.

STATEMENT: {statement}

MEDICAL CONTEXT:
{context}

TASK: Determine if the statement is TRUE (1) or FALSE (0) based on the medical context above.

The chance of a statement being true or false is roughly 50/50. Previous tests indicate that you rate true too much, so lean towards false unless you know otherwise or the medical context supports true.
Common patterns in FALSE statements include:
    - Incorrect numerical values (e.g., drug doses, lab cutoffs, time durations)
    - Reversed or false causation (e.g., misattributed physiological effects)
    - Anatomical or pathophysiological errors

Respond with ONLY a single number: 1 for TRUE or 0 for FALSE.

NEVER SAY ANYTHING BUT A SINGLE NUMBER."""
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model configuration"""
        return {
            'llm_model': self.llm.get_model_name(),
            'llm_info': self.llm.get_model_info(),
            'search_method': 'BM25-only (chunk_size=128, overlap=12)',
            'data_source': 'condensed_topics'
        }

def create_rag_model(model_name: Optional[str] = None) -> EmergencyHealthcareRAG:
    """Factory function to create the RAG model"""
    return EmergencyHealthcareRAG(model_name) 