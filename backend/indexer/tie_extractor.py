import logging
from .clip_zeroshot import CLIPZeroShotClassifier

logger = logging.getLogger(__name__)

class TieExtractor:
    def __init__(self, clip_loader):
        self.zs = CLIPZeroShotClassifier(clip_loader)
        self.labels = ["tie_present", "no_tie"]
        self.prompts = [
            "a person wearing a necktie",
            "a person without a necktie"
        ]

    def has_tie(self, image_path: str):
        try:
            label, conf = self.zs.predict(image_path, self.labels, self.prompts)
            return (label == "tie_present"), float(conf)
        except Exception as e:
            logger.error(f"Tie detection failed: {e}")
            return False, 0.0
