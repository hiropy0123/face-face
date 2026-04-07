"""DeepFace を使った顔属性推定ラッパ."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class FaceResult:
    region: dict[str, int]  # {"x": int, "y": int, "w": int, "h": int}
    age: int
    gender: str
    gender_confidence: float
    dominant_race: str
    race: dict[str, float] = field(default_factory=dict)


def _to_face_result(item: dict[str, Any]) -> FaceResult:
    region = item.get("region", {}) or {}
    gender_dict = item.get("gender", {}) or {}
    dominant_gender = item.get("dominant_gender", "")
    if isinstance(gender_dict, dict) and dominant_gender:
        gender_conf = float(gender_dict.get(dominant_gender, 0.0))
    else:
        gender_conf = 0.0

    race_dict = item.get("race", {}) or {}
    race_norm = {k: float(v) for k, v in race_dict.items()} if isinstance(race_dict, dict) else {}

    return FaceResult(
        region={
            "x": int(region.get("x", 0)),
            "y": int(region.get("y", 0)),
            "w": int(region.get("w", 0)),
            "h": int(region.get("h", 0)),
        },
        age=int(round(float(item.get("age", 0)))),
        gender=str(dominant_gender),
        gender_confidence=gender_conf,
        dominant_race=str(item.get("dominant_race", "")),
        race=race_norm,
    )


def analyze_image(image_rgb: np.ndarray) -> list[FaceResult]:
    """RGB 画像 (np.ndarray) を受け取り、検出された全顔の属性を返す."""
    # 遅延 import で Streamlit 起動を高速化
    from deepface import DeepFace

    # DeepFace は BGR 想定なので変換
    image_bgr = image_rgb[:, :, ::-1].copy()

    results = DeepFace.analyze(
        img_path=image_bgr,
        actions=["age", "gender", "race"],
        detector_backend="opencv",
        enforce_detection=False,
        silent=True,
    )

    if isinstance(results, dict):
        results = [results]

    faces: list[FaceResult] = []
    for item in results:
        fr = _to_face_result(item)
        # 顔未検出時は w=0 / h=0 になることがあるので除外
        if fr.region["w"] > 0 and fr.region["h"] > 0:
            faces.append(fr)
    return faces
