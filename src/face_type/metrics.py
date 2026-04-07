"""顔の特徴量を算出."""
import math
from dataclasses import dataclass, asdict

import cv2
import numpy as np

from .constants import LANDMARKS, JAW_LINE_INDICES


@dataclass
class FaceMetrics:
    # 基本比率
    face_width: float
    face_height: float
    aspect_ratio: float

    # パーツ位置
    eye_position_ratio: float
    forehead_ratio: float
    lower_face_ratio: float

    # パーツサイズ
    eye_width_ratio: float
    eye_height_ratio: float
    nose_length_ratio: float
    mouth_width_ratio: float
    eyebrow_eye_distance: float

    # 形状
    jaw_angle: float
    face_roundness: float
    eye_roundness: float
    eyebrow_curvature: float
    nose_width_ratio: float

    # 診断スコア（classify で書き込まれる）
    maturity_score: float = 0.0
    linearity_score: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def _dist(p1, p2) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def _midpoint(p1, p2):
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)


def _angle(p1, p2, p3) -> float:
    """p2 を頂点とした角度(度)."""
    v1 = (p1[0] - p2[0], p1[1] - p2[1])
    v2 = (p3[0] - p2[0], p3[1] - p2[1])
    dot  = v1[0]*v2[0] + v1[1]*v2[1]
    mag1 = math.hypot(*v1)
    mag2 = math.hypot(*v2)
    if mag1 * mag2 == 0:
        return 0.0
    cos_a = max(-1.0, min(1.0, dot / (mag1 * mag2)))
    return math.degrees(math.acos(cos_a))


def _curvature(p1, p2, p3) -> float:
    """3点から曲率を算出."""
    area  = abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))
    d1 = _dist(p1, p2)
    d2 = _dist(p2, p3)
    d3 = _dist(p1, p3)
    denom = d1 * d2 * d3
    if denom == 0:
        return 0.0
    return (4 * area) / denom


def extract_metrics(landmarks: list, img_w: int, img_h: int) -> FaceMetrics:
    def pt(idx):
        lm = landmarks[idx]
        return (lm.x * img_w, lm.y * img_h)

    jaw_l    = pt(LANDMARKS["jaw_left"])
    jaw_r    = pt(LANDMARKS["jaw_right"])
    chin     = pt(LANDMARKS["chin"])
    forehead = pt(LANDMARKS["forehead_top"])

    face_width  = _dist(jaw_l, jaw_r)
    face_height = _dist(forehead, chin)
    aspect_ratio = face_height / face_width if face_width > 0 else 1.0

    left_eye_center  = _midpoint(pt(LANDMARKS["left_eye_inner"]),  pt(LANDMARKS["left_eye_outer"]))
    right_eye_center = _midpoint(pt(LANDMARKS["right_eye_inner"]), pt(LANDMARKS["right_eye_outer"]))
    eye_center_y = (left_eye_center[1] + right_eye_center[1]) / 2
    eye_position_ratio = (eye_center_y - forehead[1]) / face_height if face_height > 0 else 0.5
    forehead_ratio     = eye_position_ratio
    # 下顔面比: 鼻先〜顎先 / 顔高さ（三分割法の下顔面に対応、典型値 0.28〜0.40）
    nose_tip_pt        = pt(LANDMARKS["nose_tip"])
    lower_face_ratio   = (chin[1] - nose_tip_pt[1]) / face_height if face_height > 0 else 0.33

    left_eye_w  = _dist(pt(LANDMARKS["left_eye_inner"]),  pt(LANDMARKS["left_eye_outer"]))
    left_eye_h  = _dist(pt(LANDMARKS["left_eye_top"]),    pt(LANDMARKS["left_eye_bottom"]))
    right_eye_w = _dist(pt(LANDMARKS["right_eye_inner"]), pt(LANDMARKS["right_eye_outer"]))
    right_eye_h = _dist(pt(LANDMARKS["right_eye_top"]),   pt(LANDMARKS["right_eye_bottom"]))
    avg_eye_w   = (left_eye_w + right_eye_w) / 2
    avg_eye_h   = (left_eye_h + right_eye_h) / 2
    eye_width_ratio  = avg_eye_w / face_width if face_width > 0 else 0.0
    eye_height_ratio = avg_eye_h / avg_eye_w  if avg_eye_w  > 0 else 0.0
    eye_roundness    = eye_height_ratio

    nose_length       = _dist(pt(LANDMARKS["nose_top"]),  pt(LANDMARKS["nose_tip"]))
    nose_length_ratio = nose_length / face_height if face_height > 0 else 0.0
    nose_w            = _dist(pt(LANDMARKS["nose_left"]), pt(LANDMARKS["nose_right"]))
    nose_width_ratio  = nose_w / face_width if face_width > 0 else 0.0

    mouth_w          = _dist(pt(LANDMARKS["mouth_left"]), pt(LANDMARKS["mouth_right"]))
    mouth_width_ratio = mouth_w / face_width if face_width > 0 else 0.0

    brow_eye_dist        = _dist(pt(LANDMARKS["left_eyebrow_inner"]), pt(LANDMARKS["left_eye_top"]))
    eyebrow_eye_distance = brow_eye_dist / avg_eye_h if avg_eye_h > 0 else 1.0

    jaw_points = [pt(i) for i in JAW_LINE_INDICES]
    chin_idx   = len(jaw_points) // 2
    jaw_angle  = _angle(
        jaw_points[chin_idx - 3],
        jaw_points[chin_idx],
        jaw_points[chin_idx + 3],
    )

    jaw_np = np.array(jaw_points, dtype=np.float32).reshape(-1, 1, 2)
    if len(jaw_np) >= 5:
        ellipse       = cv2.fitEllipse(jaw_np)
        axes          = ellipse[1]
        face_roundness = min(axes) / max(axes) if max(axes) > 0 else 0.5
    else:
        face_roundness = 0.5

    eyebrow_curvature = (
        _curvature(
            pt(LANDMARKS["left_eyebrow_inner"]),
            pt(LANDMARKS["left_eyebrow_peak"]),
            pt(LANDMARKS["left_eyebrow_outer"]),
        )
        + _curvature(
            pt(LANDMARKS["right_eyebrow_inner"]),
            pt(LANDMARKS["right_eyebrow_peak"]),
            pt(LANDMARKS["right_eyebrow_outer"]),
        )
    ) / 2

    return FaceMetrics(
        face_width=face_width,
        face_height=face_height,
        aspect_ratio=aspect_ratio,
        eye_position_ratio=eye_position_ratio,
        forehead_ratio=forehead_ratio,
        lower_face_ratio=lower_face_ratio,
        eye_width_ratio=eye_width_ratio,
        eye_height_ratio=eye_height_ratio,
        nose_length_ratio=nose_length_ratio,
        mouth_width_ratio=mouth_width_ratio,
        eyebrow_eye_distance=eyebrow_eye_distance,
        jaw_angle=jaw_angle,
        face_roundness=face_roundness,
        eye_roundness=eye_roundness,
        eyebrow_curvature=eyebrow_curvature,
        nose_width_ratio=nose_width_ratio,
    )
