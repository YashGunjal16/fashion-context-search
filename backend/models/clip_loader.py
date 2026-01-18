"""
CLIP model loader for vision embeddings.
Handles lazy loading and caching of CLIP model.
"""

import torch
import logging
from typing import Optional, List, Tuple
import hashlib

logger = logging.getLogger(__name__)

class CLIPLoader:
    """Manages CLIP model loading and inference."""
    
    _instance = None
    _model = None
    _processor = None
    _device = None
    
    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super(CLIPLoader, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize CLIP loader."""
        if self._model is None:
            self._initialize()
    
    def _initialize(self):
        """Lazy initialize CLIP model and processor."""
        try:
            from transformers import CLIPProcessor, CLIPModel
            
            logger.info("Loading CLIP model...")
            
            # Determine device
            self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logger.info(f"Using device: {self._device}")
            
            # Load model and processor
            model_name = "openai/clip-vit-base-patch32"
            self._model = CLIPModel.from_pretrained(model_name).to(self._device)
            self._processor = CLIPProcessor.from_pretrained(model_name)
            
            # Freeze model
            self._model.eval()
            for param in self._model.parameters():
                param.requires_grad = False
            
            logger.info("CLIP model loaded successfully")
            
        except ImportError as e:
            logger.error(f"CLIP dependencies not installed: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            raise
    
    @property
    def model(self):
        """Get CLIP model instance."""
        if self._model is None:
            self._initialize()
        return self._model
    
    @property
    def processor(self):
        """Get CLIP processor instance."""
        if self._processor is None:
            self._initialize()
        return self._processor
    
    @property
    def device(self):
        """Get computation device."""
        if self._device is None:
            self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return self._device
    
    @torch.no_grad()
    def encode_image(self, image_path: str) -> torch.Tensor:
        """
        Encode image to CLIP embedding.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Image embedding (512-dim vector for ViT-B/32)
        """
        try:
            from PIL import Image
            
            image = Image.open(image_path).convert('RGB')
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            outputs = self.model.get_image_features(**inputs)
            
            # Normalize embedding
            embedding = outputs / outputs.norm(dim=1, keepdim=True)
            
            return embedding.cpu()
            
        except Exception as e:
            logger.error(f"Failed to encode image {image_path}: {e}")
            raise
    
    @torch.no_grad()
    def encode_text(self, text: str) -> torch.Tensor:
        """
        Encode text to CLIP embedding.
        
        Args:
            text: Text to encode
            
        Returns:
            Text embedding (512-dim vector for ViT-B/32)
        """
        try:
            inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)
            outputs = self.model.get_text_features(**inputs)
            
            # Normalize embedding
            embedding = outputs / outputs.norm(dim=1, keepdim=True)
            
            return embedding.cpu()
            
        except Exception as e:
            logger.error(f"Failed to encode text '{text}': {e}")
            raise
    
    @torch.no_grad()
    def encode_batch_images(self, image_paths: List[str]) -> torch.Tensor:
        """
        Batch encode images for efficiency.
        
        Args:
            image_paths: List of image paths
            
        Returns:
            Batch embeddings (N x 512)
        """
        try:
            from PIL import Image
            
            images = [Image.open(path).convert('RGB') for path in image_paths]
            inputs = self.processor(images=images, return_tensors="pt", padding=True).to(self.device)
            outputs = self.model.get_image_features(**inputs)
            
            # Normalize embeddings
            embeddings = outputs / outputs.norm(dim=1, keepdim=True)
            
            return embeddings.cpu()
            
        except Exception as e:
            logger.error(f"Batch encoding failed: {e}")
            raise
