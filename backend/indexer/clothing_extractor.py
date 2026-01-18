import logging
from .clip_zeroshot import CLIPZeroShotClassifier

logger = logging.getLogger(__name__)

class ClothingExtractor:
    """
    Clothing type via CLIP zero-shot prompts (expanded).
    """

    def __init__(self, clip_loader):
        self.zs = CLIPZeroShotClassifier(clip_loader)

        self.labels = [
            "suit",
            "blazer",
            "coat",
            "jacket",
            "raincoat",
            "hoodie",
            "sweater",
            "button_down",
            "t_shirt",
            "dress",
            "skirt",
            "jeans",
            "pants",
            "tie",
        ]

        self.prompts = [
            "a person wearing a suit",
            "a person wearing a blazer",
            "a person wearing a long coat",
            "a person wearing a jacket",
            "a person wearing a raincoat",
            "a person wearing a hoodie",
            "a person wearing a sweater",
            "a person wearing a button-down shirt",
            "a person wearing a t-shirt",
            "a person wearing a dress",
            "a person wearing a skirt",
            "a person wearing jeans",
            "a person wearing trousers or pants",
            "a person wearing a tie",
        ]

    def extract_clothing(self, image_path: str) -> str:
        try:
            label, conf = self.zs.predict(image_path, self.labels, self.prompts)
            return label
        except Exception as e:
            logger.error(f"Clothing extraction failed: {e}")
            return "unknown"
