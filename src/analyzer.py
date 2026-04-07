"""顔属性推定ラッパ.

年齢推定戦略:
- DeepFace で全顔の race / gender / 初期 age を取得する。
- dominant_race == "asian" と判定された顔については InsightFace の
  age/gender モデル (buffalo_l) で年齢を再推定し、精度を向上させる。
  InsightFace の年齢推定 MAE は同条件で DeepFace より約 3 年低い
  (arXiv:2511.14689: DeepFace 10.83 年 vs InsightFace 7.46 年)。
- 2 つの検出器の bbox は IoU でマッチングする。IoU < 0.3 の場合は
  対応付け失敗とみなし DeepFace の値をそのまま使用する。
"""
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


# InsightFace アプリのモジュールレベルキャッシュ（遅延初期化）
_insight_app = None


def _get_insight_app():
    global _insight_app
    if _insight_app is None:
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=0, det_size=(640, 640))
        _insight_app = app
    return _insight_app


def init_insightface() -> None:
    """InsightFace モデルを事前ロードする（warmup 用）."""
    _get_insight_app()


def _iou(region: dict[str, int], bbox: list[float]) -> float:
    """DeepFace の region と InsightFace の bbox の IoU を計算する.

    Args:
        region: DeepFace の {"x": x, "y": y, "w": w, "h": h}
        bbox:   InsightFace の [x1, y1, x2, y2]
    """
    ax1 = region["x"]
    ay1 = region["y"]
    ax2 = ax1 + region["w"]
    ay2 = ay1 + region["h"]
    bx1, by1, bx2, by2 = bbox

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = region["w"] * region["h"]
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _insightface_detections(image_bgr: np.ndarray) -> list[tuple[list[float], int]]:
    """InsightFace で顔を検出し (bbox, age) のリストを返す.

    Returns:
        [(bbox [x1,y1,x2,y2], age), ...]
    """
    app = _get_insight_app()
    faces = app.get(image_bgr)
    result: list[tuple[list[float], int]] = []
    for face in faces:
        if hasattr(face, "age") and face.age is not None:
            bbox = face.bbox.tolist()  # [x1, y1, x2, y2]
            result.append((bbox, int(round(float(face.age)))))
    return result


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
    from deepface import DeepFace

    # DeepFace は BGR 想定なので変換
    image_bgr = image_rgb[:, :, ::-1].copy()

    results = DeepFace.analyze(
        img_path=image_bgr,
        actions=["age", "gender", "race"],
        detector_backend="retinaface",
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

    # アジア系と判別された顔は InsightFace で年齢を再推定する
    asian_faces = [f for f in faces if f.dominant_race == "asian"]
    if asian_faces:
        try:
            insight_detections = _insightface_detections(image_bgr)
            for face in asian_faces:
                best_iou = 0.0
                best_age: int | None = None
                for bbox, age in insight_detections:
                    iou = _iou(face.region, bbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_age = age
                # IoU 0.3 以上で同一顔と判定して年齢を置き換える
                if best_age is not None and best_iou >= 0.3:
                    face.age = best_age
        except Exception:
            # InsightFace が失敗した場合は DeepFace の年齢をそのまま使用
            pass

    return faces
