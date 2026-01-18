"""
Places365 scene classifier (vision-only) to predict environment / place.
Used to improve "where" understanding beyond CLIP.
"""

import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
from pathlib import Path


class Places365Loader:
    def __init__(self, ckpt_dir: str = "checkpoints"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ckpt_dir = Path(ckpt_dir)
        weights_path = ckpt_dir / "places365_resnet18.pth"
        labels_path = ckpt_dir / "categories_places365.txt"

        if not weights_path.exists():
            raise FileNotFoundError(f"Missing Places365 weights: {weights_path}")
        if not labels_path.exists():
            raise FileNotFoundError(f"Missing Places365 labels: {labels_path}")

        # ResNet18 trained on Places365 has 365 classes
        self.model = models.resnet18(num_classes=365)
        checkpoint = torch.load(weights_path, map_location=self.device)

        # Some checkpoints store under ["state_dict"]
        state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
        # Remove "module." prefix if present
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval().to(self.device)

        # Load label names
        with open(labels_path, "r", encoding="utf-8") as f:
            # e.g. "/a/airfield 0"
            self.classes = [line.strip().split(" ")[0][3:] for line in f]

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # Map fine scene -> coarse environment label set
        self.scene_map = {
    "office": [
        "office", "conference_room", "classroom",
        "library", "corridor", "lobby", "reception"
    ],
    "home": [
        "living_room", "bedroom", "kitchen",
        "dining_room", "closet", "home"
    ],
    "park": [
        "park", "garden", "forest", "field",
        "botanical_garden", "lawn"
    ],
    "street": [
        "street", "road", "plaza", "sidewalk",
        "downtown", "market", "shopfront", "city"
    ],
    "runway": [
        "runway", "catwalk", "stage",
        "auditorium", "theater", "museum"
    ],
    "mall_store": [
        "clothing_store", "boutique",
        "shopping_mall", "department_store"
    ]
}


    @torch.no_grad()
    def predict_fine_scene(self, image_path: str) -> str:
        img = Image.open(image_path).convert("RGB")
        x = self.transform(img).unsqueeze(0).to(self.device)
        logits = self.model(x)
        top_idx = int(torch.argmax(logits, dim=1).item())
        return self.classes[top_idx]

        # coarse mapping
        for coarse, keys in self.scene_map.items():
            for k in keys:
                if k in fine_scene:
                    return coarse

        return "unknown"
    def predict_environment(self, image_path: str) -> str:
        """
        Map Places365 fine-grained scene to coarse environment label.
        """
        fine_scene = self.predict_fine_scene(image_path).lower()

        for coarse, keywords in self.scene_map.items():
            for kw in keywords:
                if kw in fine_scene:
                    return coarse

        return "unknown"
