import torch
from PIL import Image


class CLIPZeroShotClassifier:
    """
    Zero-shot image classification using CLIP via prompt matching.
    """

    def __init__(self, clip_loader):
        """
        clip_loader must expose:
        - model (CLIPModel)
        - processor (CLIPProcessor)
        - device (torch device)
        """
        self.model = clip_loader.model
        self.processor = clip_loader.processor
        self.device = getattr(clip_loader, "device", "cpu")

    def predict(self, image_path: str, labels: list[str], prompts: list[str]):
        """
        Returns:
            best_label (str), confidence (float)
        """
        image = Image.open(image_path).convert("RGB")

        inputs = self.processor(
            text=prompts,
            images=image,
            return_tensors="pt",
            padding=True
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = outputs.logits_per_image[0].softmax(dim=0)

        best_idx = int(torch.argmax(probs).item())
        return labels[best_idx], float(probs[best_idx].item())
