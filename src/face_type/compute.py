"""MediaPipe ランドマークから FaceMetrics を計算するモジュール.

MediaPipe FaceLandmarker (478点モデル) の主要インデックス:
  10  : 額トップ
  152 : 顎先 (chin)
  234 : 左頬 (視聴者側の左)
  454 : 右頬 (視聴者側の右)
  左目(視聴者左): outer=33, inner=133, top=159, bottom=145
  右目(視聴者右): outer=263, inner=362, top=386, bottom=374
  左眉: 70, 63, 105, 66, 107
  右眉: 300, 293, 334, 296, 336
  鼻ブリッジ: 168  / 鼻先: 1
  左顎角: 172 / 右顎角: 397
"""
from __future__ import annotations

import numpy as np

from .metrics import FaceMetrics


def _xy(landmarks, index: int) -> np.ndarray:
    lm = landmarks[index]
    return np.array([lm.x, lm.y], dtype=float)


def compute_metrics(landmarks, image_width: int, image_height: int) -> FaceMetrics:
    """MediaPipe の normalized landmarks から FaceMetrics を計算する.

    landmarks: FaceLandmarker が返す face_landmarks[0]  (NormalizedLandmark のリスト)
    image_width / image_height: 元画像のピクセルサイズ (角度計算に使用)
    """
    def lm(i: int) -> np.ndarray:
        return _xy(landmarks, i)

    # --- 顔全体の境界点 ---
    face_top    = lm(10)
    face_bottom = lm(152)
    face_left   = lm(234)
    face_right  = lm(454)

    face_h = abs(face_bottom[1] - face_top[1])
    face_w = abs(face_right[0] - face_left[0])
    eps = 1e-6

    # 1. 顔の縦横比 (高さ / 幅)
    aspect_ratio = face_h / (face_w + eps)

    # 2. 目の縦位置比
    left_eye_center  = (lm(33) + lm(133)) / 2
    right_eye_center = (lm(263) + lm(362)) / 2
    eye_mid_y = (left_eye_center[1] + right_eye_center[1]) / 2
    eye_position_ratio = (eye_mid_y - face_top[1]) / (face_h + eps)

    # 3. 目幅比 (片目幅平均 / 顔幅)
    left_eye_w  = abs(lm(133)[0] - lm(33)[0])
    right_eye_w = abs(lm(263)[0] - lm(362)[0])
    eye_width_ratio = ((left_eye_w + right_eye_w) / 2) / (face_w + eps)

    # 4. 鼻の長さ比
    nose_bridge = lm(168)
    nose_tip    = lm(1)
    nose_length = abs(nose_tip[1] - nose_bridge[1])
    nose_length_ratio = nose_length / (face_h + eps)

    # 5. 下顔面比 (鼻先〜顎 / 顔高さ)
    lower_face_ratio = abs(face_bottom[1] - nose_tip[1]) / (face_h + eps)

    # 6. 顎の角度 (chin 点での内角)
    chin      = lm(152)
    left_jaw  = lm(172)
    right_jaw = lm(397)
    # normalized → pixel 座標系に変換して角度を計算
    scale = np.array([image_width, image_height], dtype=float)
    v1 = (left_jaw  - chin) * scale
    v2 = (right_jaw - chin) * scale
    norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if norm1 > eps and norm2 > eps:
        cos_a = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
        jaw_angle = float(np.degrees(np.arccos(cos_a)))
    else:
        jaw_angle = 120.0

    # 7. 顔の丸み (幅 / 高さ ; 1.0 = 正方形に近い = 丸顔)
    face_roundness = face_w / (face_h + eps)

    # 8. 目の丸み (目高さ / 目幅)
    #    高さはピクセル比を維持するため scale で補正
    left_eye_h  = abs(lm(159)[1] - lm(145)[1])
    right_eye_h = abs(lm(386)[1] - lm(374)[1])
    avg_eye_h_px = ((left_eye_h + right_eye_h) / 2) * image_height
    avg_eye_w_px = ((left_eye_w + right_eye_w) / 2) * image_width
    eye_roundness = avg_eye_h_px / (avg_eye_w_px + eps)

    # 9. 眉の曲率 (弦からの最大垂直偏差 / 弦長)
    def _brow_curvature(indices: list[int]) -> float:
        pts = np.array([lm(i) * scale for i in indices])
        p0, p1 = pts[0], pts[-1]
        chord = p1 - p0
        chord_len = np.linalg.norm(chord) + eps
        direction = chord / chord_len
        normal = np.array([-direction[1], direction[0]])
        deviations = [abs(float(np.dot(pts[j] - p0, normal))) for j in range(len(pts))]
        return max(deviations) / chord_len

    left_brow_indices  = [70, 63, 105, 66, 107]
    right_brow_indices = [300, 293, 334, 296, 336]
    eyebrow_curvature = (
        _brow_curvature(left_brow_indices) + _brow_curvature(right_brow_indices)
    ) / 2

    return FaceMetrics(
        aspect_ratio=float(aspect_ratio),
        eye_position_ratio=float(eye_position_ratio),
        eye_width_ratio=float(eye_width_ratio),
        nose_length_ratio=float(nose_length_ratio),
        lower_face_ratio=float(lower_face_ratio),
        jaw_angle=float(jaw_angle),
        face_roundness=float(face_roundness),
        eye_roundness=float(eye_roundness),
        eyebrow_curvature=float(eyebrow_curvature),
    )
