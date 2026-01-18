"""
Pydantic schemas for request/response validation.
Keeps schema layer PURE (no imports from routes/main to avoid circular deps).
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class QueryRequest(BaseModel):
    """Request schema for fashion search queries."""
    query: str = Field(..., description="User's natural language query")
    top_k: int = Field(8, ge=1, le=100, description="Number of final results to return")
    candidate_k: int = Field(50, ge=10, le=200, description="Number of candidates retrieved before reranking")
    use_llm: bool = Field(True, description="Enable LLM parser (Gemini). If low confidence, falls back to rules.")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "A model walking on a runway wearing black outfit",
                "top_k": 8,
                "candidate_k": 50,
                "use_llm": True
            }
        }


class ImageResult(BaseModel):
    """Single image result with metadata."""
    image_id: str = Field(..., description="Unique image identifier")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Final relevance score after reranking")
    clothing_type: Optional[str] = Field(None, description="Detected clothing type")
    colors: List[str] = Field(default_factory=list, description="Dominant detected colors")
    environment: Optional[str] = Field(None, description="Scene/environment category")
    vibe: Optional[str] = Field(None, description="Style/vibe category")
    explanation: str = Field(..., description="Why this image matches the query")
    image_path: Optional[str] = Field(None, description="Local image path (frontend can load from this)")


class SearchResponse(BaseModel):
    """Response schema for search queries."""
    status: str = Field(..., description="success or error")
    query: str = Field(..., description="Original query")
    query_parsed: Dict[str, Any] = Field(..., description="Parsed attributes from query understanding")
    llm_confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score returned by query parser")
    fallback_used: bool = Field(False, description="Whether rule-based fallback was used")
    processing_time: float = Field(..., description="Seconds taken end-to-end")
    results: List[ImageResult] = Field(default_factory=list, description="Ranked results")


class ErrorResponse(BaseModel):
    """Standard error response schema."""
    status: str = Field("error", description="Error status")
    error_code: str = Field(..., description="Machine-readable error code")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Extra debug details")
