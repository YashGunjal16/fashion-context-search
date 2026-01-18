"""
LLM-based query parser using Google Gemini.
Handles semantic understanding of fashion queries.
"""

import os
import json
import re
import logging
from typing import Dict, Tuple, Any

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)


class GeminiParser:
    """LLM-based query parser using Google Gemini."""

    def __init__(self):
        self.client = self._init_client()

    # --------------------------------------------------
    # CLIENT INIT
    # --------------------------------------------------

    def _init_client(self):
        try:
            import google.generativeai as genai

            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                logger.warning("GOOGLE_API_KEY not set")
                return None

            genai.configure(api_key=api_key)
            logger.info("Gemini API client initialized")
            return genai

        except ImportError:
            logger.error("google-generativeai not installed")
            return None
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}")
            return None

    # --------------------------------------------------
    # PUBLIC API
    # --------------------------------------------------

    def parse(self, query: str) -> Tuple[Dict[str, Any], float]:
        if not self.client:
            raise RuntimeError("Gemini API not available")

        try:
            prompt = self._build_prompt(query)
            model = self.client.GenerativeModel("gemini-2.5-flash")

            response = model.generate_content(prompt)
            raw_text = response.text or ""

            parsed = self._extract_json(raw_text)
            parsed = self._sanitize(parsed)

            logger.info(f"LLM parsed output: {parsed}")
            return parsed, parsed["confidence"]

        except Exception as e:
            logger.error(f"Gemini parsing failed: {e}")
            raise

    # --------------------------------------------------
    # PROMPT (UPDATED)
    # --------------------------------------------------

    def _build_prompt(self, query: str) -> str:
        clothing_items = [
            "suit", "blazer", "coat", "jacket", "raincoat", "hoodie",
            "sweater", "shirt", "button_down", "t_shirt", "dress",
            "skirt", "jeans", "pants", "shorts", "tie", "scarf"
        ]

        color_labels = [
            "red", "blue", "green", "yellow", "black",
            "white", "pink", "purple", "brown", "gray", "beige"
        ]

        env_labels = [
            "office", "home", "park", "street", "runway",
            "indoor_event", "restaurant_cafe", "mall_store"
        ]

        vibe_labels = [
            "business_formal", "business_casual", "smart_casual",
            "casual", "streetwear", "sporty_athleisure",
            "outerwear_winter", "party_evening", "runway_editorial"
        ]

        return f"""
You are a strict JSON parser for fashion search queries.

Query:
"{query}"

Return ONLY valid JSON with EXACTLY these keys:

{{
  "clothing_type": string or null,
  "colors": array of strings,

  "upper_item": string or null,
  "upper_colors": array of strings,

  "lower_item": string or null,
  "lower_colors": array of strings,

  "neck_item": string or null,
  "neck_colors": array of strings,

  "inner_item": string or null,
  "inner_colors": array of strings,

  "environment": string or null,
  "vibe": string or null,
  "confidence": number from 0.0 to 1.0
}}

Allowed values:
- clothing items: {clothing_items}
- colors: subset of {color_labels}
- environment: {env_labels}
- vibe: {vibe_labels}

Rules:
- Normalize colors (e.g. navy → blue, cream → white, beige → brown)
- If unsure, use null (or [] for arrays)
- Do NOT invent attributes
- confidence reflects certainty across ALL fields
- Return JSON only (no markdown, no text)

"""

    # --------------------------------------------------
    # SANITIZATION (CRITICAL)
    # --------------------------------------------------

    def _sanitize(self, parsed: Dict[str, Any]) -> Dict[str, Any]:
        allowed_items = {
            "suit", "blazer", "coat", "jacket", "raincoat", "hoodie",
            "sweater", "shirt", "button_down", "t_shirt", "dress",
            "skirt", "jeans", "pants", "shorts", "tie", "scarf"
        }

        allowed_colors = {
            "red", "blue", "green", "yellow", "black",
            "white", "pink", "purple", "brown", "gray", "beige"
        }

        allowed_env = {
            "office", "home", "park", "street", "runway",
            "indoor_event", "restaurant_cafe", "mall_store"
        }

        allowed_vibe = {
            "business_formal", "business_casual", "smart_casual",
            "casual", "streetwear", "sporty_athleisure",
            "outerwear_winter", "party_evening", "runway_editorial"
        }

        def clean_item(val):
            return val if val in allowed_items else None

        def clean_colors(vals):
            if not isinstance(vals, list):
                return []
            return [c for c in vals if c in allowed_colors]

        out = {
            "clothing_type": clean_item(parsed.get("clothing_type")),
            "colors": clean_colors(parsed.get("colors")),

            "upper_item": clean_item(parsed.get("upper_item")),
            "upper_colors": clean_colors(parsed.get("upper_colors")),

            "lower_item": clean_item(parsed.get("lower_item")),
            "lower_colors": clean_colors(parsed.get("lower_colors")),

            "neck_item": clean_item(parsed.get("neck_item")),
            "neck_colors": clean_colors(parsed.get("neck_colors")),

            "inner_item": clean_item(parsed.get("inner_item")),
            "inner_colors": clean_colors(parsed.get("inner_colors")),

            "environment": parsed.get("environment") if parsed.get("environment") in allowed_env else None,
            "vibe": parsed.get("vibe") if parsed.get("vibe") in allowed_vibe else None,
            "confidence": parsed.get("confidence", 0.7),
        }

        try:
            out["confidence"] = float(out["confidence"])
        except Exception:
            out["confidence"] = 0.6

        out["confidence"] = max(0.0, min(1.0, out["confidence"]))
        return out

    # --------------------------------------------------
    # JSON EXTRACTION
    # --------------------------------------------------

    def _extract_json(self, text: str) -> Dict[str, Any]:
        text = text.strip()
        text = re.sub(r"^```json", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"^```", "", text).strip()
        text = re.sub(r"```$", "", text).strip()

        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        logger.warning("Could not parse JSON from LLM response")
        return {
            "clothing_type": None,
            "colors": [],

            "upper_item": None,
            "upper_colors": [],
            "lower_item": None,
            "lower_colors": [],
            "neck_item": None,
            "neck_colors": [],
            "inner_item": None,
            "inner_colors": [],

            "environment": None,
            "vibe": None,
            "confidence": 0.5
        }
