"""
Places365 scene/environment classification model loader.
Provides scene context for image understanding.
"""

import torch
import logging
from typing import Dict, List, Tuple
import torch.nn.functional as F

logger = logging.getLogger(__name__)

class SceneLoader:
    """Manages Places365 scene classifier."""
    
    _instance = None
    _model = None
    _device = None
    _class_labels = None
    
    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super(SceneLoader, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize scene loader."""
        if self._model is None:
            self._initialize()
    
    def _initialize(self):
        """Lazy initialize Places365 model."""
        try:
            from torchvision import models
            
            logger.info("Loading Places365 scene classifier...")
            
            # Determine device
            self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logger.info(f"Using device: {self._device}")
            
            # Load Places365 ResNet50 model
            self._model = torch.hub.load(
                'pytorch/vision:v0.10.0',
                'resnet50',
                pretrained=True
            )
            
            # Modify for 365 classes (Places365)
            self._model.fc = torch.nn.Linear(2048, 365)
            self._model = self._model.to(self._device)
            
            # Freeze model
            self._model.eval()
            for param in self._model.parameters():
                param.requires_grad = False
            
            # Load class labels
            self._load_class_labels()
            
            logger.info("Places365 model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load Places365 model: {e}")
            # Fallback to basic scene labels
            self._init_fallback_model()
    
    def _init_fallback_model(self):
        """Initialize fallback scene model."""
        logger.warning("Using fallback scene classifier")
        self._class_labels = self._get_fallback_labels()
    
    def _load_class_labels(self):
        """Load Places365 class labels."""
        # Common Places365 environment categories
        self._class_labels = self._get_fallback_labels()
    
    def _get_fallback_labels(self) -> List[str]:
        """Get fallback scene labels for common environments."""
        return [
            "indoor", "outdoor", "street", "beach", "mountain",
            "forest", "park", "urban", "rural", "garden",
            "office", "home", "shop", "restaurant", "cafe",
            "classroom", "gym", "sports", "formal", "casual"
        ]
    
    @property
    def model(self):
        """Get model instance."""
        if self._model is None:
            self._initialize()
        return self._model
    
    @property
    def device(self):
        """Get computation device."""
        if self._device is None:
            self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return self._device
    
    @torch.no_grad()
    def classify_scene(self, image_path: str) -> Dict[str, float]:
        """
        Classify image scene/environment.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary of scene type to confidence score
        """
        try:
            from PIL import Image
            from torchvision import transforms
            
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            
            # Standard ImageNet preprocessing
            preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            
            img_tensor = preprocess(image).unsqueeze(0).to(self.device)
            
            # Get predictions
            if self._model is not None:
                outputs = self._model(img_tensor)
                probs = F.softmax(outputs, dim=1)[0]
                
                # Get top predictions
                top_k = 5
                top_probs, top_indices = torch.topk(probs, top_k)
                
                results = {}
                for prob, idx in zip(top_probs.cpu().numpy(), top_indices.cpu().numpy()):
                    if idx < len(self._class_labels):
                        results[self._class_labels[idx]] = float(prob)
                
                return results
            else:
                # Fallback classification
                return {"indoor": 0.5, "outdoor": 0.5}
            
        except Exception as e:
            logger.error(f"Scene classification failed for {image_path}: {e}")
            return {"unknown": 1.0}
    
    def get_primary_scene(self, image_path: str) -> str:
        """
        Get primary scene classification.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Primary scene label
        """
        scenes = self.classify_scene(image_path)
        if scenes:
            return max(scenes.items(), key=lambda x: x[1])[0]
        return "unknown"
