"""顔特徴量データクラス."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FaceMetrics:
    # --- 基本形状 ---
    aspect_ratio: float = 0.0       # 顔の縦横比 (高さ / 幅)  高い=面長
    eye_position_ratio: float = 0.0 # 目の縦位置 (顔トップ〜目中点 / 顔高さ)  低い=目が上
    eye_width_ratio: float = 0.0    # 目幅比 (片目幅平均 / 顔幅)  大きい=大きな目
    nose_length_ratio: float = 0.0  # 鼻の長さ比 (鼻ブリッジ〜先 / 顔高さ)
    lower_face_ratio: float = 0.0   # 下顔面比 (鼻先〜顎 / 顔高さ)  大きい=下が長い

    # --- 輪郭・形状 ---
    jaw_angle: float = 120.0        # 顎の角度 (°)  小さい=シャープ
    face_roundness: float = 0.75    # 顔の丸み (幅/高さ)  1.0=正円
    eye_roundness: float = 0.30     # 目の丸み (目高さ/目幅)  大きい=丸い目
    eyebrow_curvature: float = 0.010  # 眉の曲率  大きい=アーチ型

    # --- スコア (classify で書き込まれる) ---
    maturity_score: float = 0.0     # -1(幼顔) 〜 +1(大人顔)
    linearity_score: float = 0.0    # -1(曲線) 〜 +1(直線)
