"""2軸スコア算出 & 8タイプ分類（日本人顔基準）.

成熟度スコア: 大人顔(+1) ← 0 → 幼顔(-1)
直線性スコア: 直線系(+1) ← 0 → 曲線系(-1)

各閾値は日本人顔の統計的分布に基づき調整。
"""
from .constants import FaceType, MATURITY_THRESHOLD, LINEARITY_THRESHOLD
from .metrics import FaceMetrics


def _compute_maturity(m: FaceMetrics, sensitivity: float = 1.0) -> float:
    """日本人顔基準の成熟度スコアを算出する.

    日本人の平均的特徴:
    - aspect_ratio: 1.20〜1.35 (欧米人より丸め)
    - eye_position_ratio: 0.40〜0.46 (目が顔の中心〜やや上)
    - eye_width_ratio: 0.24〜0.30 (比較的大きい目)
    - nose_length_ratio: 0.24〜0.32
    - lower_face_ratio: 0.48〜0.58
    """
    score = 0.0

    # 顔の縦横比: 面長=大人顔
    if m.aspect_ratio > 1.35:
        score += 0.3
    elif m.aspect_ratio < 1.20:
        score -= 0.3

    # 目の縦位置: 高い(下に位置)=大人顔、低い=幼顔
    if m.eye_position_ratio < 0.40:
        score -= 0.2
    elif m.eye_position_ratio > 0.46:
        score += 0.2

    # 目の幅: 大きい目=幼顔
    if m.eye_width_ratio > 0.28:
        score -= 0.25
    elif m.eye_width_ratio < 0.23:
        score += 0.25

    # 鼻の長さ: 長い=大人顔
    if m.nose_length_ratio > 0.30:
        score += 0.15
    elif m.nose_length_ratio < 0.24:
        score -= 0.15

    # 下顔面比: 大きい=大人顔
    if m.lower_face_ratio > 0.55:
        score += 0.15
    elif m.lower_face_ratio < 0.48:
        score -= 0.15

    return max(-1.0, min(1.0, score * sensitivity))


def _compute_linearity(m: FaceMetrics, sensitivity: float = 1.0) -> float:
    """日本人顔基準の直線性スコアを算出する.

    日本人の平均的特徴:
    - jaw_angle: 115〜130 (欧米人よりやや丸め)
    - face_roundness: 0.72〜0.88
    - eye_roundness: 0.22〜0.35
    - eyebrow_curvature: 0.008〜0.018
    """
    score = 0.0

    # 顎の角度: 鋭い=直線系（日本人の平均は120°前後）
    if m.jaw_angle < 112:
        score += 0.3
    elif m.jaw_angle > 128:
        score -= 0.3

    # 顔の丸み: 丸い=曲線系
    if m.face_roundness > 0.82:
        score -= 0.25
    elif m.face_roundness < 0.68:
        score += 0.25

    # 目の丸み: 丸い=曲線系
    if m.eye_roundness > 0.32:
        score -= 0.2
    elif m.eye_roundness < 0.22:
        score += 0.2

    # 眉の曲率: アーチ型=曲線系
    if m.eyebrow_curvature > 0.016:
        score -= 0.15
    elif m.eyebrow_curvature < 0.007:
        score += 0.15

    return max(-1.0, min(1.0, score * sensitivity))


def classify(m: FaceMetrics, sensitivity: float = 1.0) -> tuple[FaceType, FaceMetrics]:
    """特徴量から2軸スコアを算出し、8タイプに分類して metrics に書き込む."""
    maturity  = _compute_maturity(m, sensitivity)
    linearity = _compute_linearity(m, sensitivity)
    m.maturity_score  = maturity
    m.linearity_score = linearity

    mt = MATURITY_THRESHOLD
    lt = LINEARITY_THRESHOLD

    if maturity < -mt:
        if linearity < -lt:
            face_type = FaceType.CUTE
        elif linearity > lt:
            face_type = FaceType.ACTIVE_CUTE
        else:
            face_type = FaceType.FRESH
    elif maturity > mt:
        if linearity < -lt:
            face_type = FaceType.FEMININE
        elif linearity > lt:
            face_type = FaceType.COOL
        else:
            face_type = FaceType.SOFT_ELEGANT
    else:
        if linearity < -lt / 2:
            face_type = FaceType.SOFT_ELEGANT
        elif linearity > lt / 2:
            face_type = FaceType.COOL_CASUAL
        else:
            face_type = FaceType.ELEGANT

    return face_type, m
