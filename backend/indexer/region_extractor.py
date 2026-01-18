"""
Hybrid region extractor:
- YOLO detects person
- crop to person
- split into upper/lower
- extract neck/chest band (tie/scarf zone) using:
  (a) fixed band prior
  (b) refine using edge-density scan for best band location
"""

from typing import Dict, Optional, Tuple
import cv2
import numpy as np
from ultralytics import YOLO


class RegionExtractor:
    def __init__(self, model_name: str = "yolov8n.pt", conf: float = 0.25):
        self.model = YOLO(model_name)
        self.conf = conf

    def extract_regions(self, image_path: str) -> Dict[str, Optional[np.ndarray]]:
        img = cv2.imread(image_path)
        if img is None:
            return {"person_crop": None, "upper_crop": None, "lower_crop": None, "neck_crop": None}

        h, w = img.shape[:2]
        results = self.model.predict(source=img, conf=self.conf, verbose=False)

        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return self._fallback_center_crop(img)

        boxes = results[0].boxes
        cls = boxes.cls.cpu().numpy().astype(int)
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()

        person_idxs = np.where(cls == 0)[0]  # COCO person = 0
        if len(person_idxs) == 0:
            return self._fallback_center_crop(img)

        # best person: conf * area
        best_i, best_score = None, -1.0
        for i in person_idxs:
            x1, y1, x2, y2 = xyxy[i]
            area = max(0, x2 - x1) * max(0, y2 - y1)
            score = float(confs[i]) * float(area)
            if score > best_score:
                best_score = score
                best_i = i

        x1, y1, x2, y2 = xyxy[best_i]
        x1, y1, x2, y2 = self._clamp_box(x1, y1, x2, y2, w, h)

        person = img[y1:y2, x1:x2].copy()
        if person.size == 0:
            return self._fallback_center_crop(img)

        upper, lower = self._split_upper_lower(person)
        neck = self._extract_neck_band_hybrid(upper)

        return {"person_crop": person, "upper_crop": upper, "lower_crop": lower, "neck_crop": neck}

    def _split_upper_lower(self, person_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        ph, pw = person_bgr.shape[:2]
        split = int(ph * 0.55)
        upper = person_bgr[:split, :].copy()
        lower = person_bgr[split:, :].copy()
        return upper, lower

    def _extract_neck_band_hybrid(self, upper_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Hybrid:
        - prior band near top of upper crop
        - refine by scanning nearby bands and picking max edge density
        """
        if upper_bgr is None or upper_bgr.size == 0:
            return None

        uh, uw = upper_bgr.shape[:2]

        # (A) prior: band around 15%..35% of upper (often collar/tie)
        prior_top = int(uh * 0.12)
        prior_h = int(uh * 0.22)  # band height

        # clamp
        prior_top = max(0, min(uh - 1, prior_top))
        prior_bot = max(prior_top + 1, min(uh, prior_top + prior_h))

        # (B) refine: scan +/- 10% region and pick band with max edge density
        gray = cv2.cvtColor(upper_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 60, 140)

        band_h = max(8, int(uh * 0.18))
        scan_start = max(0, prior_top - int(uh * 0.10))
        scan_end = min(uh - band_h, prior_top + int(uh * 0.10))

        best_top = prior_top
        best_score = -1.0

        for top in range(scan_start, max(scan_start, scan_end) + 1, max(1, int(uh * 0.02))):
            band = edges[top:top + band_h, :]
            # edge density score
            score = float(np.mean(band))
            if score > best_score:
                best_score = score
                best_top = top

        neck = upper_bgr[best_top:best_top + band_h, :].copy()
        if neck.size == 0:
            neck = upper_bgr[prior_top:prior_bot, :].copy()

        return neck

    def _fallback_center_crop(self, img: np.ndarray) -> Dict[str, Optional[np.ndarray]]:
        h, w = img.shape[:2]
        side = int(min(h, w) * 0.60)
        cx, cy = w // 2, h // 2
        x1 = max(0, cx - side // 2)
        y1 = max(0, cy - side // 2)
        x2 = min(w, cx + side // 2)
        y2 = min(h, cy + side // 2)

        person = img[y1:y2, x1:x2].copy()
        if person.size == 0:
            return {"person_crop": None, "upper_crop": None, "lower_crop": None, "neck_crop": None}

        upper, lower = self._split_upper_lower(person)
        neck = self._extract_neck_band_hybrid(upper)

        return {"person_crop": person, "upper_crop": upper, "lower_crop": lower, "neck_crop": neck}

    def _clamp_box(self, x1, y1, x2, y2, w, h):
        x1 = int(max(0, min(w - 1, x1)))
        y1 = int(max(0, min(h - 1, y1)))
        x2 = int(max(0, min(w, x2)))
        y2 = int(max(0, min(h, y2)))
        if x2 <= x1: x2 = min(w, x1 + 1)
        if y2 <= y1: y2 = min(h, y1 + 1)
        return x1, y1, x2, y2
