"""
FastAPI route handlers for fashion retrieval endpoints.
"""

from fastapi import APIRouter, HTTPException, Request
import time
import logging
from typing import Optional
import os

from backend.api.schemas import QueryRequest, SearchResponse, ImageResult, ErrorResponse
from backend.retrieval.query_parser import QueryParser
from backend.retrieval.search import SemanticSearcher
from backend.retrieval.reranker import AttributeReranker

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["search"])

# Lazy-loaded singletons
_query_parser: Optional[QueryParser] = None
_searcher: Optional[SemanticSearcher] = None
_reranker: Optional[AttributeReranker] = None


def get_query_parser() -> QueryParser:
    global _query_parser
    if _query_parser is None:
        logger.info("Initializing QueryParser (LLM optional)")
        _query_parser = QueryParser(llm_threshold=0.7)
    return _query_parser


def get_searcher() -> SemanticSearcher:
    global _searcher
    if _searcher is None:
        logger.info("Initializing SemanticSearcher (loads FAISS index + CLIP)")
        _searcher = SemanticSearcher()
    return _searcher


def get_reranker() -> AttributeReranker:
    global _reranker
    if _reranker is None:
        logger.info("Initializing AttributeReranker")
        _reranker = AttributeReranker()
    return _reranker


def build_image_url(request: Request, image_path: str | None) -> str | None:
    """
    Convert local image path to public URL:
    /data/raw/img_0001.jpg → http://localhost:8000/images/img_0001.jpg
    """
    if not image_path:
        return None

    filename = os.path.basename(image_path)
    base = str(request.base_url).rstrip("/")
    return f"{base}/images/{filename}"


@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Search for fashion images",
    description="Hybrid query understanding + semantic search + attribute reranking."
)
async def search(request: Request, payload: QueryRequest) -> SearchResponse:
    start = time.time()

    try:
        logger.info(f"Query: {payload.query}")

        # 1) Query understanding
        parser = get_query_parser()
        parsed, confidence, fallback = parser.parse(payload.query, use_llm=payload.use_llm)

        # 2) Semantic recall
        searcher = get_searcher()
        candidates = searcher.search(
            query=payload.query,
            parsed_attrs=parsed,
            top_k=max(payload.candidate_k, payload.top_k)
        )

        # 3) Attribute reranking
        reranker = get_reranker()
        final = reranker.rerank(
            results=candidates,
            parsed_attrs=parsed,
            top_k=payload.top_k
        )

        # 4) Format response
        results = [
            ImageResult(
                image_id=r.get("image_id", ""),
                relevance_score=float(r.get("score", 0.0)),
                clothing_type=r.get("clothing_type"),
                colors=r.get("colors", []),
                environment=r.get("environment"),
                vibe=r.get("vibe"),
                explanation=r.get("explanation", "Semantic match"),
                image_path=build_image_url(request, r.get("image_path"))
            )
            for r in final
        ]

        return SearchResponse(
            status="success",
            query=payload.query,
            query_parsed=parsed,
            llm_confidence=float(confidence),
            fallback_used=bool(fallback),
            processing_time=round(time.time() - start, 3),
            results=results
        )

    except Exception as e:
        logger.exception("Search failed")
        raise HTTPException(
            status_code=500,
            detail=ErrorResponse(
                error_code="SEARCH_FAILED",
                message=str(e),
                details={"hint": "Check FAISS index path and model initialization logs."}
            ).dict()
        )


@router.get("/status", summary="Get system status")
async def status():
    return {
        "status": "operational",
        "pipeline": [
            "QueryParser (Gemini optional -> rules fallback)",
            "SemanticSearcher (CLIP text -> FAISS recall)",
            "AttributeReranker (boost by colors/clothing/env/vibe)"
        ],
        "vector_store": "FAISS",
        "models": {
            "vision": "CLIP ViT-B/32 (frozen)",
            "llm": "Google Gemini (optional)"
        }
    }
