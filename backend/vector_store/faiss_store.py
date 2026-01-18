"""
FAISS vector store for efficient similarity search.
Manages indexing and retrieval of image embeddings.
"""

import faiss
import numpy as np
import logging
import os
import json
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)

class FAISSStore:
    """FAISS-based vector store for image embeddings."""
    
    def __init__(self, embedding_dim: int = 512, index_path: Optional[str] = None):
        """
        Initialize FAISS vector store.
        
        Args:
            embedding_dim: Dimension of embeddings (512 for CLIP ViT-B/32)
            index_path: Optional path to load existing index
        """
        self.embedding_dim = embedding_dim
        self.index = None
        self.metadata = []  # List of metadata dicts for each vector
        self.id_to_metadata = {}  # Map image IDs to metadata
        
        if index_path and os.path.exists(index_path):
            self.load_index(index_path)
        else:
            self._init_index()
    
    def _init_index(self):
        """Initialize FAISS index."""
        # Use IVFFlat for efficient similarity search
        quantizer = faiss.IndexFlatL2(self.embedding_dim)
        self.index = faiss.IndexIVFFlat(
            quantizer,
            self.embedding_dim,
            min(100, self.embedding_dim),  # Number of clusters
            faiss.METRIC_L2
        )
        logger.info(f"Initialized FAISS index with dimension {self.embedding_dim}")
    
    def add_embedding(
        self,
        image_id: str,
        embedding: np.ndarray,
        metadata: Dict
    ):
        """
        Add single embedding to index.
        
        Args:
            image_id: Unique image identifier
            embedding: Image embedding (1D array of size 512)
            metadata: Associated metadata (clothing type, colors, etc.)
        """
        if embedding.shape[0] != self.embedding_dim:
            raise ValueError(f"Embedding dimension mismatch: {embedding.shape[0]} vs {self.embedding_dim}")
        
        # Add to FAISS index
        embedding = embedding.reshape(1, -1).astype(np.float32)
        if not self.index.is_trained:
            self.index.train(embedding)
        
        self.index.add(embedding)
        
        # Store metadata
        metadata['image_id'] = image_id
        self.metadata.append(metadata)
        self.id_to_metadata[image_id] = metadata
        
        logger.debug(f"Added embedding for {image_id}")
    
    def add_batch_embeddings(
        self,
        embeddings: np.ndarray,
        metadatas: List[Dict]
    ):
        """
        Batch add embeddings for efficiency.
        
        Args:
            embeddings: Batch of embeddings (N x 512)
            metadatas: List of metadata dicts
        """
        if embeddings.shape[0] != len(metadatas):
            raise ValueError("Embeddings and metadatas length mismatch")
        
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(f"Embedding dimension mismatch: {embeddings.shape[1]} vs {self.embedding_dim}")
        
        # Ensure float32
        embeddings = embeddings.astype(np.float32)
        
        # Train index if needed
        if not self.index.is_trained:
            self.index.train(embeddings)
        
        # Add to index
        self.index.add(embeddings)
        
        # Store metadata
        for i, metadata in enumerate(metadatas):
            self.metadata.append(metadata)
            image_id = metadata.get('image_id', f'img_{len(self.metadata)-1}')
            self.id_to_metadata[image_id] = metadata
        
        logger.info(f"Added batch of {len(embeddings)} embeddings")
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10
    ) -> List[Dict]:
        """
        Search for similar embeddings.
        
        Args:
            query_embedding: Query embedding (1D array of size 512)
            top_k: Number of results to return
            
        Returns:
            List of result dicts with scores and metadata
        """
        if not self.index.is_trained or self.index.ntotal == 0:
            logger.warning("Index is empty or not trained")
            return []
        
        query = query_embedding.reshape(1, -1).astype(np.float32)
        
        # Search
        distances, indices = self.index.search(query, min(top_k, self.index.ntotal))
        
        # Format results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0 and idx < len(self.metadata):
                result = self.metadata[idx].copy()
                # Convert L2 distance to similarity score (0-1)
                # Assuming embeddings are normalized
                similarity = 1.0 / (1.0 + float(dist))
                result['score'] = similarity
                results.append(result)
        
        return results
    
    def save_index(self, path: str):
        """
        Save index to disk.
        
        Args:
            path: Directory path to save index
        """
        os.makedirs(path, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, os.path.join(path, 'index.faiss'))
        
        # Save metadata
        metadata_path = os.path.join(path, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        logger.info(f"Index saved to {path}")
    
    def load_index(self, path: str):
        """
        Load index from disk.
        
        Args:
            path: Directory path containing index files
        """
        # Load FAISS index
        index_path = os.path.join(path, 'index.faiss')
        self.index = faiss.read_index(index_path)

        if hasattr(self.index, "nprobe"):
            self.index.nprobe = 10  # good default for ~1k–10k images
        
        # Load metadata
        metadata_path = os.path.join(path, 'metadata.json')
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        # Rebuild ID mapping
        self.id_to_metadata = {m['image_id']: m for m in self.metadata}
        
        logger.info(f"Index loaded from {path}")
    
    def get_size(self) -> int:
        """Get number of embeddings in index."""
        if self.index:
            return self.index.ntotal
        return 0
