import logging
from .clip_zeroshot import CLIPZeroShotClassifier

logger = logging.getLogger(__name__)

class VibeExtractor:
    """
    Vibe/style inference via CLIP prompts (expanded).
    """

    def __init__(self, clip_loader):
        self.zs = CLIPZeroShotClassifier(clip_loader)

        self.labels = [
            "business_formal",
            "business_casual",
            "smart_casual",
            "casual",
            "streetwear",
            "sporty_athleisure",
            "outerwear_winter",
            "party_evening",
            "runway_editorial",
        ]

        self.prompts = [
            "a formal business outfit, professional attire",
            "a business casual outfit suitable for office",
            "a smart casual outfit",
            "a casual everyday outfit",
            "a streetwear outfit",
            "a sporty athleisure outfit",
            "a winter outerwear outfit with coat or jacket",
            "a party evening outfit, dressy look",
            "a runway editorial fashion look on a catwalk",
        ]

    def extract_vibe(self, image_path: str) -> str:
        try:
            label, conf = self.zs.predict(image_path, self.labels, self.prompts)
            return label
        except Exception as e:
            logger.error(f"Vibe extraction failed: {e}")
            return "casual"
