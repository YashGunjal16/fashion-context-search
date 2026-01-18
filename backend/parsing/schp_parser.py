import os
import sys
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms


class SCHPParser:
    """
    SCHP (LIP) single-image parser:
    returns label_map (H, W) where each pixel is a class id.
    We use it only to get upper/lower clothing masks.
    """

    def __init__(
        self,
        schp_root="external/schp",
        ckpt_path="checkpoints/schp/schp_lip.pth",
        dataset="lip"
    ):
        if not os.path.exists(schp_root):
            raise FileNotFoundError(f"SCHP not found: {schp_root}")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"SCHP checkpoint not found: {ckpt_path}")

        self.schp_root = schp_root
        self.ckpt_path = ckpt_path
        self.dataset = dataset

        # IMPORTANT: make SCHP imports work
        sys.path.insert(0, schp_root)

        from simple_extractor import dataset_settings
        import networks

        self.dataset_settings = dataset_settings
        self.input_size = self.dataset_settings[self.dataset]["input_size"]
        self.num_classes = self.dataset_settings[self.dataset]["num_classes"]

        # Build model exactly like their script
        self.model = networks.init_model(
            "resnet101",
            num_classes=self.num_classes,
            pretrained=None
        )

        ckpt = torch.load(self.ckpt_path, map_location="cpu")
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt

        # remove "module." prefix (same logic as their script)
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            nk = k[7:] if k.startswith("module.") else k
            new_state_dict[nk] = v

        # strict=False because we patched BN (cpu-safe)
        self.model.load_state_dict(new_state_dict, strict=False)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        # same normalization (note: their script uses mean/std in reverse order, we keep it same as them)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.406, 0.456, 0.485],
                std=[0.225, 0.224, 0.229]
            ),
        ])

        # transform helper from SCHP
        from utils.transforms import transform_logits
        self.transform_logits = transform_logits

    @torch.no_grad()
    def parse_label_map(self, image_path: str) -> np.ndarray:
        """
        GOLD-STANDARD SCHP inference (LIP)
        Returns label_map (H, W) aligned to original image.
        """
        from PIL import Image
        import torch.nn.functional as F

        # --------------------------------------------------
        # 1. Load image (RGB)
        # --------------------------------------------------
        img = Image.open(image_path).convert("RGB")
        w, h = img.size

        # --------------------------------------------------
        # 2. Resize EXACTLY like SCHP (short side -> input_size)
        # --------------------------------------------------
        input_w, input_h = self.input_size

        scale = max(input_w / w, input_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        img_resized = img.resize((new_w, new_h), Image.BILINEAR)

        # Center crop
        left = (new_w - input_w) // 2
        top = (new_h - input_h) // 2
        img_crop = img_resized.crop((left, top, left + input_w, top + input_h))

        # --------------------------------------------------
        # 3. Normalize
        # --------------------------------------------------
        x = self.transform(img_crop).unsqueeze(0).to(self.device)

        # --------------------------------------------------
        # 4. Forward pass
        # --------------------------------------------------
        output = self.model(x)

        # Official SCHP output format
        logits = output[0][-1][0]   # (C, H, W)

        # --------------------------------------------------
        # 5. Upsample logits back to cropped image
        # --------------------------------------------------
        logits = logits.unsqueeze(0)
        logits = F.interpolate(
            logits,
            size=(input_h, input_w),
            mode="bilinear",
            align_corners=True
        )[0]

        # --------------------------------------------------
        # 6. Place logits back into resized image
        # --------------------------------------------------
        full_logits = torch.zeros(
            (logits.shape[0], new_h, new_w),
            device=logits.device
        )
        full_logits[:, top:top + input_h, left:left + input_w] = logits

        # --------------------------------------------------
        # 7. Resize back to original image size
        # --------------------------------------------------
        full_logits = F.interpolate(
            full_logits.unsqueeze(0),
            size=(h, w),
            mode="bilinear",
            align_corners=True
        )[0]

        # --------------------------------------------------
        # 8. Argmax → label map
        # --------------------------------------------------
        label_map = torch.argmax(full_logits, dim=0).cpu().numpy().astype(np.uint8)

        return label_map

    def upper_lower_masks(self, label_map: np.ndarray):
        """
        LIP labels:
          5  Upper-clothes
          6  Dress
          7  Coat
          9  Pants
          12 Skirt
        """
        upper_ids = {5, 6, 7}
        lower_ids = {6, 9, 12}   # dress counts as both

        upper_mask = np.isin(label_map, list(upper_ids))
        lower_mask = np.isin(label_map, list(lower_ids))

        return upper_mask, lower_mask
