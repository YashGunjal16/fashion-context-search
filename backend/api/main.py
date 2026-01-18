"""
FastAPI backend entrypoint for Fashion Context Search.
Run:
  uvicorn backend.api.main:app --reload
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import logging
import os

from backend.api.routes import router

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_ROOT = os.path.abspath(os.path.join(BASE_DIR, "../../data/raw"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Fashion Context Search API (models load lazily on first request)")
    yield
    logger.info("Shutting down Fashion Context Search API")


app = FastAPI(
    title="Fashion Context Search API",
    version="1.0.0",
    description="Multimodal fashion & context retrieval using CLIP + FAISS + hybrid query parsing",
    lifespan=lifespan
)

# Serve images
app.mount("/images", StaticFiles(directory=IMAGE_ROOT), name="images")

# CORS (Streamlit frontend will call this)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later for deployment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(router)


@app.get("/", tags=["Health"])
async def root():
    return {
        "service": "fashion-context-search",
        "status": "ok",
        "docs": "/docs",
        "search_endpoint": "/api/search"
    }


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "healthy"}
