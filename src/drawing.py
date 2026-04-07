"""検出結果の描画ユーティリティ."""
from __future__ import annotations

import cv2
import numpy as np

from .analyzer import FaceResult

BOX_COLOR = (0, 200, 0)       # RGB green
TEXT_COLOR = (255, 255, 255)  # white
TEXT_BG = (0, 200, 0)


def draw_faces(image_rgb: np.ndarray, faces: list[FaceResult]) -> np.ndarray:
    """RGB 画像に bounding box とラベルを描画して返す."""
    img = image_rgb.copy()
    for idx, face in enumerate(faces, start=1):
        x, y, w, h = face.region["x"], face.region["y"], face.region["w"], face.region["h"]
        cv2.rectangle(img, (x, y), (x + w, y + h), BOX_COLOR, 2)

        label = f"#{idx} Age:{face.age} {face.gender} {face.dominant_race}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.5, min(w, h) / 200.0)
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(label, font, scale, thickness)

        ty = max(y - 6, th + 4)
        cv2.rectangle(
            img,
            (x, ty - th - 4),
            (x + tw + 4, ty + baseline - 2),
            TEXT_BG,
            -1,
        )
        cv2.putText(img, label, (x + 2, ty - 2), font, scale, TEXT_COLOR, thickness, cv2.LINE_AA)
    return img


def crop_face(image_rgb: np.ndarray, face: FaceResult, padding: float = 0.1) -> np.ndarray:
    """顔領域を切り出す(パディング付き)."""
    h_img, w_img = image_rgb.shape[:2]
    x, y, w, h = face.region["x"], face.region["y"], face.region["w"], face.region["h"]
    pad_x = int(w * padding)
    pad_y = int(h * padding)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(w_img, x + w + pad_x)
    y2 = min(h_img, y + h + pad_y)
    return image_rgb[y1:y2, x1:x2]
