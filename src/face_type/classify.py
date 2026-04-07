"""2軸スコア算出 & 8タイプ分類（日本人顔統計基準）.

========== 参照データ ==========

【顔の縦横比 (aspect_ratio = 顔高さ / 顔幅)】
  産業技術総合研究所「日本人頭部寸法データベース2001」
    女性: 21.8cm / 13.8cm ≈ 1.58
    男性: 23.2cm / 14.5cm ≈ 1.60
  ※ MediaPipe は額〜顎を顔高さとするため、
    実測より短くなる。実測値は1.35〜1.50 が日本人の典型範囲。
    幼顔(丸顔)寄り: < 1.30
    大人顔(面長)寄り: > 1.42

【目の幅比 (eye_width_ratio = 片目幅 / 顔幅)】
  花王「日本人女性の平均顔解析」(2021)
    目の横幅 : 顔幅 ≈ 1 : 5 → 0.20 が平均
    幼顔(大きな目)寄り: > 0.23
    大人顔(小さめの目)寄り: < 0.18

【目の縦位置 (eye_position_ratio = 顔上端〜目中点 / 顔高さ)】
  一般的な顔面計測では目は顔高さの 40〜45% に位置する。
    幼顔(目が下寄り): < 0.41
    大人顔(目が上寄り): > 0.45

【鼻の長さ比 (nose_length_ratio = 鼻ブリッジ〜鼻先 / 顔高さ)】
  顔の三分割比では中顔面(目〜鼻先)が顔高さの 1/3 ≈ 0.33。
  鼻自体の長さは中顔面の 70〜80% 程度。
    平均: 0.24〜0.32
    大人顔(長い鼻): > 0.30
    幼顔(短い鼻)  : < 0.23

【下顔面比 (lower_face_ratio = 鼻先〜顎 / 顔高さ)】
  令和時代の理想比「上:中:下 = 1:1:0.8」に基づき
  下顔面が縮小傾向にある。
    平均: 0.45〜0.55
    大人顔(長い下顔面): > 0.53
    幼顔(短い下顔面) : < 0.47

【顎の角度 (jaw_angle)】
  「日本人乾燥頭蓋骨の下顎形態統計研究」
    日本人平均: 約120°（欧米人より約5° 鋭い傾向）
    直線系(シャープな顎): < 114°
    曲線系(丸い顎)      : > 126°

【顔の丸み (face_roundness = 輪郭楕円の短軸/長軸)】
  日本人は欧米人と比べてやや丸め。
    平均: 0.70〜0.84
    曲線系(丸顔): > 0.80
    直線系(細面): < 0.67

【目の丸み (eye_roundness = 目高さ / 目幅)】
  日本人は一重・奥二重が多く、目の開き比が小さい傾向。
    平均: 0.20〜0.30
    曲線系(丸い目): > 0.30
    直線系(細い目): < 0.20

【眉の曲率 (eyebrow_curvature)】
  アーチ型眉(曲線系): > 0.015
  フラット眉(直線系): < 0.007

================================
"""
from __future__ import annotations

from .constants import FaceType, LINEARITY_THRESHOLD, MATURITY_THRESHOLD
from .metrics import FaceMetrics


def _compute_maturity(m: FaceMetrics, sensitivity: float = 1.0) -> float:
    """成熟度スコアを算出する: 大人顔 = +1.0 / 幼顔 = -1.0.

    各特徴の「日本人平均」からのズレがどちら方向かで加減算する。
    """
    score = 0.0

    # ── 顔の縦横比: 面長=大人顔, 丸顔=幼顔 ──
    # 日本人平均: 1.35〜1.42
    if m.aspect_ratio > 1.42:
        score += 0.30
    elif m.aspect_ratio > 1.35:
        score += 0.10
    elif m.aspect_ratio < 1.25:
        score -= 0.30
    elif m.aspect_ratio < 1.32:
        score -= 0.10

    # ── 目の縦位置: 高い位置=大人顔, 低い位置=幼顔 ──
    # 日本人平均: 0.41〜0.45
    if m.eye_position_ratio > 0.45:
        score += 0.20
    elif m.eye_position_ratio < 0.40:
        score -= 0.20

    # ── 目の幅比: 小さい目=大人顔, 大きい目=幼顔 ──
    # 日本人平均: 0.18〜0.23 (顔幅の1/5)
    if m.eye_width_ratio > 0.23:
        score -= 0.25
    elif m.eye_width_ratio > 0.20:
        score -= 0.08
    elif m.eye_width_ratio < 0.17:
        score += 0.25
    elif m.eye_width_ratio < 0.20:
        score += 0.08

    # ── 鼻の長さ比: 長い鼻=大人顔 ──
    # 日本人平均: 0.24〜0.30
    if m.nose_length_ratio > 0.30:
        score += 0.15
    elif m.nose_length_ratio < 0.22:
        score -= 0.15

    # ── 下顔面比: 長い下顔面=大人顔 ──
    # 令和理想は下顔面が小さめ (0.45〜0.53 が日本人の典型範囲)
    if m.lower_face_ratio > 0.53:
        score += 0.15
    elif m.lower_face_ratio < 0.46:
        score -= 0.15

    # ── 目の丸み: 細い目=大人顔, 丸い目=幼顔 ──
    # 日本人の一重・奥二重を考慮した補正項
    if m.eye_roundness < 0.18:
        score += 0.10
    elif m.eye_roundness > 0.32:
        score -= 0.10

    return max(-1.0, min(1.0, score * sensitivity))


def _compute_linearity(m: FaceMetrics, sensitivity: float = 1.0) -> float:
    """直線性スコアを算出する: 直線系(シャープ) = +1.0 / 曲線系(丸み) = -1.0."""
    score = 0.0

    # ── 顎の角度: 鋭い顎=直線系 ──
    # 日本人平均: 約120°
    if m.jaw_angle < 112:
        score += 0.30
    elif m.jaw_angle < 117:
        score += 0.12
    elif m.jaw_angle > 128:
        score -= 0.30
    elif m.jaw_angle > 122:
        score -= 0.12

    # ── 顔の丸み: 細面=直線系, 丸顔=曲線系 ──
    # 日本人平均: 0.70〜0.84
    if m.face_roundness > 0.82:
        score -= 0.25
    elif m.face_roundness > 0.76:
        score -= 0.08
    elif m.face_roundness < 0.65:
        score += 0.25
    elif m.face_roundness < 0.71:
        score += 0.08

    # ── 目の丸み: 細い目=直線系, 丸い目=曲線系 ──
    # 日本人の一重・奥二重は eye_roundness が低め (0.20〜0.28 が典型)
    if m.eye_roundness > 0.32:
        score -= 0.20
    elif m.eye_roundness > 0.27:
        score -= 0.08
    elif m.eye_roundness < 0.18:
        score += 0.20
    elif m.eye_roundness < 0.23:
        score += 0.08

    # ── 眉の曲率: アーチ型=曲線系, フラット=直線系 ──
    if m.eyebrow_curvature > 0.016:
        score -= 0.15
    elif m.eyebrow_curvature < 0.007:
        score += 0.15

    # ── 鼻の幅: 広い鼻=曲線系 (日本人に多い) ──
    # nose_width_ratio の平均は 0.26〜0.34
    if m.nose_width_ratio > 0.36:
        score -= 0.10
    elif m.nose_width_ratio < 0.25:
        score += 0.10

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
