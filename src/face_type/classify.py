"""2軸スコア算出 & 8タイプ分類."""
from .constants import FaceType, MATURITY_THRESHOLD, LINEARITY_THRESHOLD
from .metrics import FaceMetrics


def _compute_maturity(m: FaceMetrics, sensitivity: float = 1.0) -> float:
    score = 0.0
    if m.aspect_ratio > 1.45:
        score += 0.3
    elif m.aspect_ratio < 1.30:
        score -= 0.3

    if m.eye_position_ratio < 0.42:
        score -= 0.2
    elif m.eye_position_ratio > 0.48:
        score += 0.2

    if m.eye_width_ratio > 0.26:
        score -= 0.25
    elif m.eye_width_ratio < 0.22:
        score += 0.25

    if m.nose_length_ratio > 0.30:
        score += 0.15
    elif m.nose_length_ratio < 0.25:
        score -= 0.15

    if m.lower_face_ratio > 0.55:
        score += 0.15
    elif m.lower_face_ratio < 0.48:
        score -= 0.15

    return max(-1.0, min(1.0, score * sensitivity))


def _compute_linearity(m: FaceMetrics, sensitivity: float = 1.0) -> float:
    score = 0.0
    if m.jaw_angle < 110:
        score += 0.3
    elif m.jaw_angle > 130:
        score -= 0.3

    if m.face_roundness > 0.85:
        score -= 0.25
    elif m.face_roundness < 0.70:
        score += 0.25

    if m.eye_roundness > 0.35:
        score -= 0.2
    elif m.eye_roundness < 0.25:
        score += 0.2

    if m.eyebrow_curvature > 0.015:
        score -= 0.15
    elif m.eyebrow_curvature < 0.008:
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
