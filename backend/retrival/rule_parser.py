"""
Rule-based query parser using spaCy NLP.
Provides deterministic fallback parsing without LLM dependency.
"""

import logging
from typing import Dict, Any, List
import re

logger = logging.getLogger(__name__)

# ----------------------------
# ITEM CONSTANTS
# ----------------------------

UPPER_ITEMS = [
    "shirt", "t-shirt", "tee", "top", "blouse",
    "hoodie", "sweater", "jacket", "coat", "blazer", "dress"
]

LOWER_ITEMS = ["pants", "jeans", "trousers", "skirt", "shorts"]
NECK_ITEMS = ["tie", "scarf"]

LAYER_WORDS = ["over", "under", "on top of"]


class RuleBasedParser:
    """Rule-based query parser with spaCy + deterministic rules."""

    def __init__(self):
        self.nlp = self._init_spacy()
        self.setup_keywords()

    # ----------------------------
    # SPAcY INIT
    # ----------------------------
    def _init_spacy(self):
        try:
            import spacy
            nlp = spacy.load("en_core_web_sm")
            logger.info("spaCy model loaded")
            return nlp
        except Exception as e:
            logger.warning(f"spaCy model not available: {e}")
            return None

    # ----------------------------
    # KEYWORDS
    # ----------------------------
    def setup_keywords(self):
        # ✅ COLOR KEYWORDS (YOUR PATTERN)
        self.color_keywords = {
            "red": ["red", "crimson", "scarlet"],
            "blue": ["blue", "navy", "azure", "cobalt"],
            "green": ["green", "emerald", "olive"],
            "yellow": ["yellow", "gold", "golden"],
            "black": ["black", "dark"],
            "white": ["white", "cream", "ivory"],
            "pink": ["pink", "rose"],
            "purple": ["purple", "violet", "lavender"],
            "brown": ["brown", "tan", "beige"],
            "gray": ["gray", "grey", "charcoal"],
        }

        self.environment_keywords = {
            "casual": ["casual", "everyday", "street", "relaxed"],
            "formal": ["formal", "professional", "business", "office"],
            "outdoor": ["outdoor", "hiking", "camping", "nature"],
            "beach": ["beach", "summer", "tropical"],
            "party": ["party", "clubbing", "night"],
            "workout": ["workout", "gym", "sports", "athletic"],
            "date": ["date", "romantic"],
        }

        self.vibe_keywords = {
            "casual": ["casual", "relaxed", "chill"],
            "elegant": ["elegant", "sophisticated", "chic"],
            "sporty": ["sporty", "athletic", "active"],
            "vintage": ["vintage", "retro"],
            "bohemian": ["bohemian", "boho", "hippie"],
            "minimalist": ["minimalist", "simple", "clean"],
            "bold": ["bold", "statement", "dramatic"],
            "classic": ["classic", "timeless", "traditional"],
        }

    # ----------------------------
    # COLOR HELPERS
    # ----------------------------
    def _extract_colors(self, text: str) -> List[str]:
        """Extract normalized colors using color_keywords."""
        found = []
        for color, keywords in self.color_keywords.items():
            for kw in keywords:
                if re.search(rf"\b{kw}\b", text) and color not in found:
                    found.append(color)
        return found

    def _find_color_item_pairs(self, text: str, items: List[str]):
        """Find (color, item) pairs using synonym-aware color matching."""
        pairs = []
        for color, keywords in self.color_keywords.items():
            for kw in keywords:
                for item in items:
                    if (
                        re.search(rf"\b{kw}\s+{item}\b", text)
                        or re.search(rf"\b{item}\s+{kw}\b", text)
                    ):
                        pairs.append((color, item))
        return pairs

    # ----------------------------
    # MAIN PARSER
    # ----------------------------
    def parse(self, query: str) -> Dict[str, Any]:
        q = query.lower()

        attrs = {
            "clothing_type": None,
            "colors": self._extract_colors(q),
            "environment": self._extract_environment(q),
            "vibe": self._extract_vibe(q),

            "upper_item": None,
            "upper_colors": [],
            "lower_item": None,
            "lower_colors": [],
            "neck_item": None,
            "neck_colors": [],
            "inner_item": None,
            "inner_colors": [],
        }

        # ----------------------------
        # NECK ITEMS
        # ----------------------------
        neck_pairs = self._find_color_item_pairs(q, NECK_ITEMS)
        if neck_pairs:
            attrs["neck_item"] = neck_pairs[0][1]
            attrs["neck_colors"] = list({p[0] for p in neck_pairs})
        else:
            for it in NECK_ITEMS:
                if re.search(rf"\b{it}\b", q):
                    attrs["neck_item"] = it

        # ----------------------------
        # UPPER ITEMS
        # ----------------------------
        upper_pairs = self._find_color_item_pairs(q, UPPER_ITEMS)
        if upper_pairs:
            attrs["upper_item"] = upper_pairs[0][1]
            attrs["upper_colors"] = list({p[0] for p in upper_pairs})

        # ----------------------------
        # LOWER ITEMS
        # ----------------------------
        lower_pairs = self._find_color_item_pairs(q, LOWER_ITEMS)
        if lower_pairs:
            attrs["lower_item"] = lower_pairs[0][1]
            attrs["lower_colors"] = list({p[0] for p in lower_pairs})

        # ----------------------------
        # LAYERING LOGIC
        # ----------------------------
        if any(w in q for w in LAYER_WORDS):
            outer_candidates = ["jacket", "coat", "blazer", "hoodie"]
            inner_candidates = ["shirt", "t-shirt", "tee", "top", "blouse", "sweater"]

            outer = next((i for i in outer_candidates if re.search(rf"\b{i}\b", q)), None)
            inner = next((i for i in inner_candidates if re.search(rf"\b{i}\b", q)), None)

            if outer:
                attrs["upper_item"] = outer

            if inner:
                attrs["inner_item"] = inner
                inner_pairs = self._find_color_item_pairs(q, [inner])
                if inner_pairs:
                    attrs["inner_colors"] = list({p[0] for p in inner_pairs})

        # ----------------------------
        # FALLBACK CLOTHING TYPE
        # ----------------------------
        if attrs["upper_item"] and attrs["upper_item"] != "dress":
            attrs["clothing_type"] = attrs["upper_item"]
        elif attrs["lower_item"]:
            attrs["clothing_type"] = attrs["lower_item"]
        elif attrs["neck_item"]:
            attrs["clothing_type"] = attrs["neck_item"]
        elif "dress" in q:
            attrs["clothing_type"] = "dress"

        return attrs

    # ----------------------------
    # CONTEXT EXTRACTION
    # ----------------------------
    def _extract_environment(self, query: str) -> str:
        for env, kws in self.environment_keywords.items():
            if any(kw in query for kw in kws):
                return env
        return None

    def _extract_vibe(self, query: str) -> str:
        for vibe, kws in self.vibe_keywords.items():
            if any(kw in query for kw in kws):
                return vibe
        return None
