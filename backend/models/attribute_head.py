"""
Attribute classification head for clothing attributes.
Fine-tuned model head for color, style, and fit predictions.
"""

import torch
import torch.nn as nn
import logging
from typing import Dict, List
import os

logger = logging.getLogger(__name__)

class AttributeHead(nn.Module):
    """Neural network head for clothing attribute prediction."""
    
    def __init__(self, input_dim: int = 512):
        """
        Initialize attribute head.
        
        Args:
            input_dim: Input embedding dimension (512 for CLIP ViT-B/32)
        """
        super().__init__()
        
        # Attribute prediction heads
        self.color_head = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 11)  # 11 color classes
        )
        
        self.clothing_type_head = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 15)  # 15 clothing type classes
        )
        
        self.style_head = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 8)  # 8 style classes
        )
        
        # Class mappings
        self.color_classes = [
            "black", "white", "red", "blue", "green",
            "yellow", "purple", "pink", "brown", "gray", "beige"
        ]
        
        self.clothing_type_classes = [
            "dress", "shirt", "pants", "jacket", "coat", "skirt",
            "sweater", "blouse", "jeans", "shorts", "hoodie",
            "cardigan", "vest", "suit", "other"
        ]
        
        self.style_classes = [
            "casual", "formal", "sporty", "vintage",
            "bohemian", "minimalist", "bold", "classic"
        ]
    
    def forward(self, embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass to predict attributes.
        
        Args:
            embeddings: Input embeddings (batch_size x 512)
            
        Returns:
            Dictionary of attribute predictions
        """
        return {
            'colors': self.color_head(embeddings),
            'clothing_types': self.clothing_type_head(embeddings),
            'styles': self.style_head(embeddings)
        }
    
    def predict_attributes(self, embeddings: torch.Tensor, threshold: float = 0.5) -> Dict[str, List[str]]:
        """
        Predict clothing attributes with confidence filtering.
        
        Args:
            embeddings: Input embeddings (batch_size x 512)
            threshold: Confidence threshold for predictions
            
        Returns:
            Dictionary of predicted attributes
        """
        with torch.no_grad():
            outputs = self.forward(embeddings)
            
            results = {}
            
            # Process color predictions
            color_probs = torch.softmax(outputs['colors'], dim=1)
            color_scores, color_indices = torch.topk(color_probs, k=3, dim=1)
            results['colors'] = [
                [self.color_classes[idx.item()] for idx in indices if color_scores[i, j] > threshold]
                for i, (scores, indices) in enumerate(zip(color_scores, color_indices))
            ]
            
            # Process clothing type predictions
            type_probs = torch.softmax(outputs['clothing_types'], dim=1)
            type_scores, type_indices = torch.topk(type_probs, k=1, dim=1)
            results['clothing_types'] = [
                self.clothing_type_classes[idx.item()] if type_scores[i, 0] > threshold else "unknown"
                for i, idx in enumerate(type_indices)
            ]
            
            # Process style predictions
            style_probs = torch.softmax(outputs['styles'], dim=1)
            style_scores, style_indices = torch.topk(style_probs, k=2, dim=1)
            results['styles'] = [
                [self.style_classes[idx.item()] for idx in indices if style_scores[i, j] > threshold]
                for i, (scores, indices) in enumerate(zip(style_scores, style_indices))
            ]
            
            return results

class AttributeHeadManager:
    """Manages loading and inference with attribute head."""
    
    _instance = None
    _model = None
    _device = None
    
    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super(AttributeHeadManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize manager."""
        if self._model is None:
            self._initialize()
    
    def _initialize(self):
        """Lazy initialize attribute head."""
        try:
            self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logger.info(f"Initializing AttributeHead on {self._device}")
            
            self._model = AttributeHead(input_dim=512).to(self._device)
            self._model.eval()
            
            # Try to load checkpoint if exists
            checkpoint_path = os.path.join(
                os.path.dirname(__file__),
                '../../checkpoints/attribute_head.pt'
            )
            
            if os.path.exists(checkpoint_path):
                logger.info(f"Loading checkpoint from {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location=self._device)
                self._model.load_state_dict(checkpoint)
                logger.info("Checkpoint loaded successfully")
            else:
                logger.warning(f"Checkpoint not found at {checkpoint_path}")
                logger.info("Using randomly initialized attribute head")
            
        except Exception as e:
            logger.error(f"Failed to initialize AttributeHead: {e}")
            raise
    
    @property
    def model(self):
        """Get model instance."""
        if self._model is None:
            self._initialize()
        return self._model
    
    @property
    def device(self):
        """Get device."""
        if self._device is None:
            self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return self._device
    
    def predict(self, embeddings: torch.Tensor) -> Dict[str, any]:
        """
        Predict attributes from embeddings.
        
        Args:
            embeddings: CLIP embeddings (N x 512)
            
        Returns:
            Attribute predictions
        """
        embeddings = embeddings.to(self.device)
        return self._model.predict_attributes(embeddings)
