"""顔類似度計算モジュール.

InsightFace の buffalo_l モデルを使い、2 枚の画像から顔を検出して類似度を算出する。

類似度の算出方法:
  1. 顔埋め込みベクトル(512 次元 ArcFace) のコサイン類似度 — 重み 70%
  2. 顔ランドマーク(5 点)から計算した幾何学的特徴の類似度 — 重み 30%
     - 目の間隔比(目間距離 / 顔幅)
     - 目-口の縦距離比(目中点〜口中点 / 顔高さ)
     - 鼻の位置比(目中点〜鼻 / 目中点〜口中点)
     - 口幅比(口幅 / 顔幅)
     - 顔の縦横比(顔高さ / 顔幅)
     - 鼻の左右オフセット比(鼻の横ずれ / 顔幅)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# 幾何学特徴ごとの感度 (この差分で 0% になる)
_SENSITIVITY: dict[str, float] = {
    "目の間隔比": 0.20,
    "目-口の縦距離比": 0.20,
    "鼻の縦位置比": 0.20,
    "口幅比": 0.20,
    "顔の縦横比": 0.40,
    "鼻の左右オフセット比": 0.12,
}


@dataclass
class FaceGeometry:
    """5-keypoint から計算した正規化済み幾何学特徴."""
    eye_spacing_ratio: float       # 目間距離 / 顔幅
    eye_mouth_ratio: float         # 目中点〜口中点 / 顔高さ
    nose_vertical_ratio: float     # 目中点〜鼻 / 目中点〜口中点
    mouth_width_ratio: float       # 口幅 / 顔幅
    face_aspect_ratio: float       # 顔高さ / 顔幅
    nose_offset_ratio: float       # 鼻横ずれ / 顔幅 (0 が正中)

    def to_dict(self) -> dict[str, float]:
        return {
            "目の間隔比": self.eye_spacing_ratio,
            "目-口の縦距離比": self.eye_mouth_ratio,
            "鼻の縦位置比": self.nose_vertical_ratio,
            "口幅比": self.mouth_width_ratio,
            "顔の縦横比": self.face_aspect_ratio,
            "鼻の左右オフセット比": self.nose_offset_ratio,
        }


@dataclass
class FaceDetection:
    """InsightFace が返す 1 顔分の情報."""
    bbox: list[float]                        # [x1, y1, x2, y2]
    kps: np.ndarray                          # shape (5, 2)
    embedding: np.ndarray                    # shape (512,)
    geometry: FaceGeometry = field(init=False)
    crop: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 3), dtype=np.uint8))

    def __post_init__(self) -> None:
        self.geometry = _compute_geometry(self.bbox, self.kps)


@dataclass
class FeatureScore:
    name: str
    value1: float
    value2: float
    score: float   # 0〜100


@dataclass
class SimilarityResult:
    overall_score: float            # 0〜100
    embedding_score: float          # 0〜100 (コサイン類似度を変換)
    geometric_score: float          # 0〜100
    feature_scores: list[FeatureScore]


# ---------------------------------------------------------------------------
# 内部ヘルパー
# ---------------------------------------------------------------------------

def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _compute_geometry(bbox: list[float], kps: np.ndarray) -> FaceGeometry:
    """bbox と 5 keypoint から正規化幾何学特徴を計算する."""
    # kps order: left_eye(0), right_eye(1), nose(2), left_mouth(3), right_mouth(4)
    left_eye   = kps[0]
    right_eye  = kps[1]
    nose       = kps[2]
    left_mouth = kps[3]
    right_mouth = kps[4]

    face_w = bbox[2] - bbox[0]
    face_h = bbox[3] - bbox[1]
    eps = 1e-6

    eye_mid   = (left_eye + right_eye) / 2
    mouth_mid = (left_mouth + right_mouth) / 2

    eye_spacing   = _dist(left_eye, right_eye) / (face_w + eps)
    eye_mouth_d   = float(abs(mouth_mid[1] - eye_mid[1])) / (face_h + eps)
    nose_v_ratio  = float(abs(nose[1] - eye_mid[1])) / (float(abs(mouth_mid[1] - eye_mid[1])) + eps)
    mouth_width   = _dist(left_mouth, right_mouth) / (face_w + eps)
    aspect_ratio  = (face_h + eps) / (face_w + eps)
    nose_offset   = float(abs(nose[0] - eye_mid[0])) / (face_w + eps)

    return FaceGeometry(
        eye_spacing_ratio=eye_spacing,
        eye_mouth_ratio=eye_mouth_d,
        nose_vertical_ratio=nose_v_ratio,
        mouth_width_ratio=mouth_width,
        face_aspect_ratio=aspect_ratio,
        nose_offset_ratio=nose_offset,
    )


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _cosine_to_score(cosine_sim: float) -> float:
    """ArcFace コサイン類似度 (-1〜1) を 0〜100 のスコアに変換する.

    - 0.6 以上: 同一人物の可能性が高い → 80〜100%
    - 0.3 前後: 微妙な類似          → 50〜65%
    - 0.0 以下: 別人               → 0〜30%
    """
    # sigmoid-like mapping centered at 0.3
    # score = 1 / (1 + exp(-k * (x - center))) * 100
    import math
    k = 8.0
    center = 0.30
    score = 1.0 / (1.0 + math.exp(-k * (cosine_sim - center))) * 100.0
    return max(0.0, min(100.0, score))


def _feature_similarity(v1: float, v2: float, sensitivity: float) -> float:
    """2 つの特徴値の類似度 (0〜100) を返す."""
    diff = abs(v1 - v2)
    return max(0.0, (1.0 - diff / sensitivity) * 100.0)


# ---------------------------------------------------------------------------
# 公開 API
# ---------------------------------------------------------------------------

def detect_face(image_rgb: np.ndarray) -> FaceDetection | None:
    """画像から最も大きな顔を 1 つ検出して返す。顔未検出なら None."""
    from src.analyzer import _get_insight_app
    from src.drawing import crop_face
    from src.analyzer import FaceResult

    image_bgr = image_rgb[:, :, ::-1].copy()
    app = _get_insight_app()
    faces = app.get(image_bgr)

    if not faces:
        return None

    # 最も大きな顔を選ぶ
    def _area(f):
        b = f.bbox
        return (b[2] - b[0]) * (b[3] - b[1])

    face = max(faces, key=_area)

    if not hasattr(face, "kps") or face.kps is None:
        return None
    if not hasattr(face, "embedding") or face.embedding is None:
        return None

    bbox = face.bbox.tolist()
    kps = np.array(face.kps)
    embedding = np.array(face.embedding)

    det = FaceDetection(bbox=bbox, kps=kps, embedding=embedding)

    # 顔クロップ画像も保存
    region = {
        "x": int(bbox[0]), "y": int(bbox[1]),
        "w": int(bbox[2] - bbox[0]), "h": int(bbox[3] - bbox[1]),
    }
    fr = FaceResult(region=region, age=0, gender="", gender_confidence=0.0, dominant_race="")
    det.crop = crop_face(image_rgb, fr, padding=0.15)
    return det


def compute_similarity(det1: FaceDetection, det2: FaceDetection) -> SimilarityResult:
    """2 つの顔検出結果から類似度を計算する."""
    # 1. 埋め込みベクトルのコサイン類似度
    cos_sim = _cosine_similarity(det1.embedding, det2.embedding)
    emb_score = _cosine_to_score(cos_sim)

    # 2. 幾何学的特徴の類似度
    g1 = det1.geometry.to_dict()
    g2 = det2.geometry.to_dict()

    feature_scores: list[FeatureScore] = []
    geo_scores: list[float] = []
    for name, sensitivity in _SENSITIVITY.items():
        v1, v2 = g1[name], g2[name]
        s = _feature_similarity(v1, v2, sensitivity)
        feature_scores.append(FeatureScore(name=name, value1=v1, value2=v2, score=s))
        geo_scores.append(s)

    geo_score = float(np.mean(geo_scores))

    # 3. 総合スコア: 埋め込み 70% + 幾何 30%
    overall = emb_score * 0.70 + geo_score * 0.30

    return SimilarityResult(
        overall_score=round(overall, 1),
        embedding_score=round(emb_score, 1),
        geometric_score=round(geo_score, 1),
        feature_scores=feature_scores,
    )
