import logging
from .clip_zeroshot import CLIPZeroShotClassifier

logger = logging.getLogger(__name__)

class EnvironmentExtractor:
    """
    Environment context via CLIP zero-shot prompts.
    Expanded label set to reduce runway->street errors and improve context retrieval.
    """

    def __init__(self, clip_loader):
        self.zs = CLIPZeroShotClassifier(clip_loader)

        self.labels = [
            "office",
            "home",
            "park",
            "street",
            "runway",
            "indoor_event",
            "restaurant_cafe",
            "mall_store",
        ]

        self.prompts = [
            "a photo taken inside a modern office with desks and computers",
            "a photo taken inside a home living room or bedroom",
            "a photo taken in a park with trees, grass, and benches",
            "a photo taken on a city street or sidewalk outdoors",
            "a fashion runway show or catwalk with a model walking",
            "an indoor public event with audience seating and stage lighting",
            "a photo taken inside a cafe or restaurant",
            "a photo taken inside a shopping mall or clothing store",
        ]

    def extract_environment(self, image_path: str) -> str:
        try:
            label, conf = self.zs.predict(image_path, self.labels, self.prompts)
            return label
        except Exception as e:
            logger.error(f"Environment extraction failed: {e}")
            return "unknown"
