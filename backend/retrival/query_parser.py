"""
Main query parsing orchestrator.
Coordinates LLM-based and rule-based parsing with confidence evaluation.
"""

import logging
from typing import Dict, Tuple, Any
from .llm_parser import GeminiParser
from .rule_parser import RuleBasedParser
from .confidence import ConfidenceEvaluator

logger = logging.getLogger(__name__)

class QueryParser:
    """Orchestrates query understanding with hybrid approach."""
    
    def __init__(self, llm_threshold: float = 0.7):
        """
        Initialize query parser.
        
        Args:
            llm_threshold: Confidence threshold for using LLM results
        """
        self.llm_threshold = llm_threshold
        self.llm_parser = GeminiParser()
        self.rule_parser = RuleBasedParser()
        self.confidence_evaluator = ConfidenceEvaluator()
    
    def parse(
        self,
        query: str,
        use_llm: bool = True
    ) -> Tuple[Dict[str, Any], float, bool]:
        """
        Parse user query with hybrid approach.
        
        Args:
            query: User's natural language query
            use_llm: Whether to use LLM parsing (fallback to rule-based if False or fails)
            
        Returns:
            Tuple of (parsed_attributes, confidence_score, fallback_used)
        """
        logger.info(f"Parsing query: {query}")
        
        fallback_used = False
        confidence = 0.0
        parsed_attrs: Dict[str, Any] = {}
        
        # Try LLM parsing first if enabled
        if use_llm:
            try:
                llm_result, llm_conf = self.llm_parser.parse(query)
                logger.info(f"LLM parsing confidence: {llm_conf:.2f}")
                
                if llm_conf >= self.llm_threshold:
                    parsed_attrs = llm_result
                    confidence = llm_conf
                    logger.info("Using LLM parsing result")
                else:
                    logger.warning(
                        f"LLM confidence {llm_conf:.2f} below threshold {self.llm_threshold}"
                    )
                    fallback_used = True
            
            except Exception as e:
                logger.error(f"LLM parsing failed: {e}. Using fallback.")
                fallback_used = True
        
        # Use rule-based parsing if needed
        if not parsed_attrs or fallback_used:
            rule_result = self.rule_parser.parse(query)
            parsed_attrs = rule_result
            confidence = self.confidence_evaluator.evaluate(
                query=query,
                parsed_attrs=rule_result,
                is_llm=False
            )
            logger.info(f"Using rule-based parsing with confidence: {confidence:.2f}")
        
        # Ensure required fields
        parsed_attrs = self._ensure_fields(parsed_attrs)

        # Normalize environment
        parsed_attrs["environment"] = self._normalize_environment(
            parsed_attrs.get("environment")
        )
        
        return parsed_attrs, confidence, fallback_used
    
    def _ensure_fields(self, attrs: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure all required fields are present."""
        default = {
            "clothing_type": None,
            "colors": [],
            "environment": None,
            "vibe": None,

            # NEW region-level fields
        "upper_item": None,
        "upper_colors": [],
        "lower_item": None,
        "lower_colors": [],
        "neck_item": None,
        "neck_colors": [],

        # Optional layering
        "inner_item": None,
        "inner_colors": []
        }
        
        for key, value in default.items():
            if key not in attrs or attrs[key] is None:
                attrs[key] = value
        
        return attrs

    def _normalize_environment(self, env: str | None) -> str | None:
        if not env:
            return None

        env = env.lower().strip()

        mapping = {
            "modern office": "office",
            "office": "office",
            "work": "office",
            "workplace": "office",

            "street": "street",
            "city": "street",
            "urban": "street",

            "park": "park",
            "garden": "park",

            "home": "home",
            "indoors": "home",
            "indoor": "home",

            "runway": "runway",
            "catwalk": "runway",

            "formal": "office",   # important: map "formal setting" -> office
            "casual": None        # don't force env on casual
        }

        return mapping.get(env, env)
