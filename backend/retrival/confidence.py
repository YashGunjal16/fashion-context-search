"""
Confidence evaluation for parsed queries.
Assesses reliability of parsing results.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class ConfidenceEvaluator:
    """Evaluates confidence of query parsing."""
    
    def evaluate(
        self,
        query: str,
        parsed_attrs: Dict[str, Any],
        is_llm: bool = False
    ) -> float:
        """
        Evaluate confidence in parsed attributes.
        
        Args:
            query: Original query
            parsed_attrs: Parsed attributes
            is_llm: Whether result came from LLM
            
        Returns:
            Confidence score (0-1)
        """
        score = 0.0
        weights = {
            'clothing_type': 0.3,
            'colors': 0.2,
            'environment': 0.25,
            'vibe': 0.25
        }
        
        # Check clothing type
        if parsed_attrs.get('clothing_type'):
            score += weights['clothing_type'] * 1.0
        
        # Check colors
        colors = parsed_attrs.get('colors', [])
        if colors:
            score += weights['colors'] * min(1.0, len(colors) / 3.0)
        
        # Check environment
        if parsed_attrs.get('environment'):
            score += weights['environment'] * 1.0
        
        # Check vibe
        if parsed_attrs.get('vibe'):
            score += weights['vibe'] * 1.0
        
        # Boost confidence for LLM results
        if is_llm:
            score = min(1.0, score * 1.1)
        
        logger.info(f"Confidence score: {score:.2f}")
        return score
