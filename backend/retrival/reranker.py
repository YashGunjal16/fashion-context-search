"""
Reranking module for refining search results.
Applies weighted attribute-based reranking (colors, clothing, environment, vibe),
with region-level compositional boosts and garment presence gating.
"""

import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class AttributeReranker:
    """Reranks search results using parsed attributes."""

    def __init__(self):
        pass

    def rerank(
        self,
        results: List[Dict[str, Any]],
        parsed_attrs: Dict[str, Any],
        top_k: int = 10
    ) -> List[Dict[str, Any]]:

        if not results:
            return []

        # --- Tuned base weights ---
        w_color = 0.10
        w_cloth = 0.22
        w_env   = 0.18
        w_vibe  = 0.10

        reranked = []

        for result in results:
            # Base CLIP semantic similarity
            base = float(result.get("score", 0.0))
            bonus = 0.0

            # -------------------------
            # GLOBAL COLOR MATCH
            # -------------------------
            if parsed_attrs.get("colors"):
                want = set(parsed_attrs["colors"])
                have = set(result.get("colors", []))
                if want:
                    color_match = len(want & have) / max(1, len(want))
                    bonus += w_color * color_match

            # -------------------------
            # CLOTHING TYPE
            # -------------------------
            if parsed_attrs.get("clothing_type"):
                if result.get("clothing_type") == parsed_attrs["clothing_type"]:
                    bonus += w_cloth

            # -------------------------
            # ENVIRONMENT
            # -------------------------
            if parsed_attrs.get("environment"):
                if result.get("environment") == parsed_attrs["environment"]:
                    bonus += w_env
                else:
                    bonus -= 0.10

            # -------------------------
            # VIBE
            # -------------------------
            if parsed_attrs.get("vibe"):
                if result.get("vibe") == parsed_attrs["vibe"]:
                    bonus += w_vibe

            # ============================================================
            # -------- REGION MATCH BOOSTS (COMPOSITIONALITY) -------------
            # ============================================================

            # Upper (shirt / jacket / top)
            if parsed_attrs.get("upper_colors"):
                want = set(parsed_attrs["upper_colors"])
                have = set(result.get("upper_colors", []))
                if want and (want & have):
                    bonus += 0.18

            # Lower (pants / skirt)
            if parsed_attrs.get("lower_colors"):
                want = set(parsed_attrs["lower_colors"])
                have = set(result.get("lower_colors", []))
                if want and (want & have):
                    bonus += 0.18

            # Neck (tie / scarf) — highest value
            if parsed_attrs.get("neck_colors") or parsed_attrs.get("neck_item") == "tie":
                tie_present = bool(result.get("tie_present", False))
                tie_conf = float(result.get("tie_conf", 0.0))

                if tie_present and tie_conf > 0.55:
                    want = set(parsed_attrs.get("neck_colors", []))
                    have = set(result.get("neck_colors", []))

                    if not want:
                        bonus += 0.18
                    elif want & have:
                        bonus += 0.25
                    else:
                        bonus -= 0.10
                else:
                    bonus -= 0.18

            # Inner layer (shirt under jacket)
            if parsed_attrs.get("inner_colors"):
                want = set(parsed_attrs["inner_colors"])
                have = set(result.get("upper_colors", []))  # approximation
                if want and (want & have):
                    bonus += 0.10

            # ============================================================
            # -------- REGION PRESENCE GATING (KEY FIX) ------------------
            # ============================================================

            result_type = result.get("clothing_type")

            # If query wants pants/jeans → penalize dresses & tops
            if parsed_attrs.get("lower_item") in {"pants", "jeans", "trousers"}:
                if result_type in {"dress"}:
                    bonus -= 0.12

            # If query wants shirt/top → penalize dresses & bottoms
            if parsed_attrs.get("upper_item") in {"shirt", "t-shirt", "tee", "button_down", "top"}:
                if result_type in {"pants", "jeans", "skirt"}:
                    bonus -= 0.12

            # -------------------------
            # FINAL SCORE
            # -------------------------
            score = base + bonus
            score = max(0.0, min(1.0, score))

            result["score"] = score
            result["explanation"] = self._generate_explanation(result, parsed_attrs)
            reranked.append(result)

        reranked.sort(key=lambda x: x["score"], reverse=True)
        return reranked[:top_k]

    # --------------------------------------------------
    # EXPLANATION GENERATOR
    # --------------------------------------------------

    def _generate_explanation(
        self,
        result: Dict[str, Any],
        query_attrs: Dict[str, Any]
    ) -> str:
        reasons = []

        # Clothing type
        if query_attrs.get("clothing_type") and result.get("clothing_type") == query_attrs["clothing_type"]:
            reasons.append(f"Matches {query_attrs['clothing_type']}")

        # Global colors
        q_colors = query_attrs.get("colors") or []
        r_colors = result.get("colors") or []
        matches = [c for c in q_colors if c in r_colors]
        if matches:
            reasons.append(f"Colors: {', '.join(matches)}")

        # -------- Region-level explanations --------
        if query_attrs.get("upper_colors"):
            reasons.append(f"Upper colors match: {', '.join(query_attrs['upper_colors'])}")

        if query_attrs.get("lower_colors"):
            reasons.append(f"Lower colors match: {', '.join(query_attrs['lower_colors'])}")

        if query_attrs.get("neck_colors"):
            reasons.append(f"Tie/neck colors match: {', '.join(query_attrs['neck_colors'])}")

        if query_attrs.get("inner_colors"):
            reasons.append(f"Inner layer hints: {', '.join(query_attrs['inner_colors'])}")

        # Vibe
        if query_attrs.get("vibe") and result.get("vibe") == query_attrs["vibe"]:
            reasons.append(f"{query_attrs['vibe'].replace('_',' ').title()} style")

        # Environment
        if query_attrs.get("environment") and result.get("environment"):
            if query_attrs["environment"] == result["environment"]:
                reasons.append(f"Scene: {result['environment']}")

        return " | ".join(reasons) if reasons else "Semantic match"
