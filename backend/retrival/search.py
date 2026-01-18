"""
Semantic search engine using CLIP embeddings and FAISS.
Performs initial retrieval from vector store.
"""

import logging
from typing import List, Dict, Any
import numpy as np
import os



from backend.models.clip_loader import CLIPLoader
from backend.vector_store.faiss_store import FAISSStore


logger = logging.getLogger(__name__)

class SemanticSearcher:
    """Performs semantic search using CLIP and FAISS."""
    
    def __init__(self):
        """Initialize semantic searcher."""
        self.clip_loader = CLIPLoader()
        self.faiss_store = FAISSStore()
        self._load_index()
    
    def _load_index(self):
    # backend/retrieval/search.py
    # __file__ -> .../fashion-context-search/backend/retrieval/search.py
        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../")
        )

        index_path = os.path.join(
            project_root,
            "data",
            "processed",
            "faiss_index"
        )

        if os.path.exists(index_path):
            self.faiss_store.load_index(index_path)
            logger.info(
                f"Loaded FAISS index with {self.faiss_store.get_size()} embeddings from {index_path}"
            )
        else:
            logger.error(
                f"FAISS index NOT FOUND at {index_path}. "
                f"Check build_index output path."
            )

    def search(
        self,
        query: str,
        parsed_attrs: Dict[str, Any],
        top_k: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic search for query.
        
        Args:
            query: Text query
            parsed_attrs: Parsed query attributes
            top_k: Number of results to retrieve
            
        Returns:
            List of retrieved results with metadata
        """
        try:
            logger.info(f"Searching for: {query}")
            
            # Encode query with CLIP
            query_embedding = self.clip_loader.encode_text(query)
            
            # Convert to numpy for FAISS
            query_vec = query_embedding.squeeze(0).numpy()
            
            # Search FAISS index
            results = self.faiss_store.search(query_vec, top_k=top_k)
            
            logger.info(f"Retrieved {len(results)} candidates")
            
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
