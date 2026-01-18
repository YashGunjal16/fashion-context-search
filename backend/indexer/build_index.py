"""
Build FAISS index from image dataset.
Creates embeddings and indexes them for fast retrieval.
"""

import logging
import os
import sys
from pathlib import Path
from typing import List, Dict
import numpy as np
import cv2

from backend.models.clip_loader import CLIPLoader
from backend.vector_store.faiss_store import FAISSStore
from backend.indexer.color_extractor import ColorExtractor
from backend.indexer.environment_extractor import EnvironmentExtractor
from backend.indexer.clothing_extractor import ClothingExtractor
from backend.indexer.vibe_extractor import VibeExtractor
from backend.models.places365_loader import Places365Loader
from backend.indexer.region_extractor import RegionExtractor
from backend.indexer.tie_extractor import TieExtractor
from backend.parsing.schp_parser import SCHPParser

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IndexBuilder:
    """Builds FAISS index from dataset."""

    def __init__(self):
        """Initialize index builder."""
        self.clip_loader = CLIPLoader()
        self.color_extractor = ColorExtractor()
        self.env_extractor = EnvironmentExtractor(self.clip_loader)
        self.clothing_extractor = ClothingExtractor(self.clip_loader)
        self.vibe_extractor = VibeExtractor(self.clip_loader)
        self.places365 = Places365Loader()
        self.region_extractor = RegionExtractor()
        self.tie_extractor = TieExtractor(self.clip_loader)
        self.schp = SCHPParser()

        self.faiss_store = FAISSStore()

    def build_index(self, image_dir: str, output_dir: str, batch_size: int = 32):
        """
        Build FAISS index from image directory.

        Args:
            image_dir: Path to directory containing images
            output_dir: Path to save index
            batch_size: Batch size for processing
        """
        logger.info(f"Building index from {image_dir}")

        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.webp'}
        image_files = [
            f for f in Path(image_dir).rglob('*')
            if f.suffix.lower() in image_extensions
        ]

        logger.info(f"Found {len(image_files)} images")

        if not image_files:
            logger.warning("No images found!")
            return

        # Process in batches
        embeddings_batch = []
        metadatas_batch = []

        for idx, image_file in enumerate(image_files):
            logger.info(f"Processing {idx + 1}/{len(image_files)}: {image_file.name}")

            try:
                # Get CLIP embedding
                embedding = self.clip_loader.encode_image(str(image_file))

                env_places = self.places365.predict_environment(str(image_file))

                regions = self.region_extractor.extract_regions(str(image_file))

                # upper_colors = self.color_extractor.extract_colors_from_bgr(regions["upper_crop"])
                # lower_colors = self.color_extractor.extract_colors_from_bgr(regions["lower_crop"])
                neck_pixels = regions["neck_crop"].reshape(-1, 3)
                neck_colors = self.color_extractor._colors_from_pixels(
                neck_pixels, num_colors=2
                )

                # tie_present, tie_conf = self.tie_extractor.has_tie(str(image_file))

                # ✅ GOLD-STANDARD BLOCK
                img_bgr = cv2.imread(str(image_file))
                if img_bgr is None:
                    logger.warning(f"Failed to read image {image_file}")
                    continue

                label_map = self.schp.parse_label_map(str(image_file))
                upper_mask, lower_mask = self.schp.upper_lower_masks(label_map)

                upper_colors = self.color_extractor.extract_colors_from_mask(
                    img_bgr, upper_mask
                )
                lower_colors = self.color_extractor.extract_colors_from_mask(
                    img_bgr, lower_mask
                )

                # 🔍 Optional sanity log (keep for now)
                if not upper_colors and not lower_colors:
                    logger.debug(
                        f"{image_file.name} mask pixels "
                        f"upper={int(upper_mask.sum())} "
                        f"lower={int(lower_mask.sum())}"
                    )

                # Extract metadata
                metadata = {
                    'image_id': image_file.stem,
                    'image_path': str(image_file),
                    'clothing_type': self.clothing_extractor.extract_clothing(str(image_file)),
                    'colors': self.color_extractor.extract_colors(str(image_file)),
                    'environment': env_places,
                    'vibe': self.vibe_extractor.extract_vibe(str(image_file)),
                    "upper_colors": upper_colors,
                    "lower_colors": lower_colors,
                    "neck_colors": neck_colors,
                    # "tie_present": tie_present,
                    # "tie_conf": tie_conf,
                }

                embeddings_batch.append(embedding.squeeze(0).numpy())
                metadatas_batch.append(metadata)

                # Add batch to index when full
                if len(embeddings_batch) >= batch_size:
                    embeddings_arr = np.array(embeddings_batch)
                    self.faiss_store.add_batch_embeddings(embeddings_arr, metadatas_batch)
                    embeddings_batch = []
                    metadatas_batch = []
                    logger.info(
                        f"Processed {idx + 1} images, index size: {self.faiss_store.get_size()}"
                    )

            except Exception as e:
                logger.error(f"Failed to process {image_file}: {e}")
                continue

        # Add remaining batch
        if embeddings_batch:
            embeddings_arr = np.array(embeddings_batch)
            self.faiss_store.add_batch_embeddings(embeddings_arr, metadatas_batch)

        # Save index
        self.faiss_store.save_index(output_dir)
        logger.info(
            f"Index built successfully. Total embeddings: {self.faiss_store.get_size()}"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build FAISS index from images")
    parser.add_argument(
        '--image_dir',
        type=str,
        default='data/raw',
        help='Directory containing images'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/processed/faiss_index',
        help='Output directory for FAISS index'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for processing'
    )

    args = parser.parse_args()

    builder = IndexBuilder()
    builder.build_index(args.image_dir, args.output_dir, args.batch_size)
