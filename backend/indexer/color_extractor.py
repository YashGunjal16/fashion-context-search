"""
Color extraction from images using clustering.
Supports extraction from full image or masked regions (upper/lower clothes).
"""

import logging
from typing import List, Dict, Tuple, Optional
import numpy as np
import cv2
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)

class ColorExtractor:
    """Extracts dominant colors from full image or masked region."""

    def __init__(self):
        # Basic palette; extend later if you want finer naming
        self.color_map = {
            "red": (255, 0, 0),
            "blue": (0, 0, 255),
            "green": (0, 255, 0),
            "yellow": (255, 255, 0),
            "black": (0, 0, 0),
            "white": (255, 255, 255),
            "pink": (255, 192, 203),
            "purple": (128, 0, 128),
            "brown": (165, 42, 42),
            "gray": (128, 128, 128),
            "beige": (245, 245, 220),
            "navy": (0, 0, 128),
        }

    def extract_colors(self, image_path: str, num_colors: int = 3) -> List[str]:
        """Extract dominant colors from the whole image."""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return []
            img = cv2.resize(img, (160, 160))
            pixels = img.reshape(-1, 3)
            return self._colors_from_pixels(pixels, num_colors=num_colors)
        except Exception as e:
            logger.error(f"Color extraction failed for {image_path}: {e}")
            return []

    def extract_colors_from_mask(
        self,
        image_bgr: np.ndarray,
        mask_hw: np.ndarray,
        num_colors: int = 3
    ) -> List[str]:
        """
        Extract dominant colors from masked region.
        mask_hw: HxW boolean or 0/1 mask
        """
        try:
            if image_bgr is None or mask_hw is None:
                return []

            # Ensure boolean mask
            mask = mask_hw.astype(bool)

            # 🔥 CRITICAL: resize mask to image size if needed
            if mask.shape[:2] != image_bgr.shape[:2]:
                mask = cv2.resize(
                    mask.astype(np.uint8),
                    (image_bgr.shape[1], image_bgr.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                ).astype(bool)

            pixels = image_bgr[mask]  # (N,3)

            # 🔥 Allow small regions
            if pixels.shape[0] < 50:
                return []

            # Subsample if huge
            if pixels.shape[0] > 20000:
                idx = np.random.choice(pixels.shape[0], 20000, replace=False)
                pixels = pixels[idx]

            return self._colors_from_pixels(pixels, num_colors=num_colors)

        except Exception as e:
            logger.error(f"Color extraction (mask) failed: {e}")
            return []

    def _colors_from_pixels(self, pixels_bgr: np.ndarray, num_colors: int = 3) -> List[str]:
        """Cluster pixels and map cluster centers to nearest named color."""
        pixels = pixels_bgr.astype(np.float32)

        k = min(num_colors, max(1, pixels.shape[0] // 200))  # adaptive clusters
        k = max(1, min(k, num_colors))

        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        kmeans.fit(pixels)

        centers = kmeans.cluster_centers_.astype(int)

        names = []
        for c in centers:
            name = self._nearest_color_name(tuple(c))
            if name not in names:
                names.append(name)

        return names[:num_colors]

    def _nearest_color_name(self, bgr: Tuple[int, int, int]) -> str:
        """Map BGR to nearest named color."""
        min_dist = float("inf")
        nearest = "gray"

        for name, rgb in self.color_map.items():
            color_bgr = (rgb[2], rgb[1], rgb[0])
            dist = np.linalg.norm(np.array(bgr) - np.array(color_bgr))
            if dist < min_dist:
                min_dist = dist
                nearest = name

        return nearest
