"""ランドマークインデックス定義・閾値・顔タイプEnum."""
from enum import Enum


class FaceType(Enum):
    CUTE         = "キュート"
    ACTIVE_CUTE  = "アクティブキュート"
    FRESH        = "フレッシュ"
    COOL_CASUAL  = "クールカジュアル"
    FEMININE     = "フェミニン"
    SOFT_ELEGANT = "ソフトエレガント"
    ELEGANT      = "エレガント"
    COOL         = "クール"


# MediaPipe Face Mesh 478点の主要インデックス
LANDMARKS = {
    # 輪郭
    "jaw_left":  234,
    "jaw_right": 454,
    "chin":      152,
    "forehead_top": 10,
    # 左目（画像上の右目）
    "left_eye_inner":  133,
    "left_eye_outer":   33,
    "left_eye_top":    159,
    "left_eye_bottom": 145,
    # 右目（画像上の左目）
    "right_eye_inner":  362,
    "right_eye_outer":  263,
    "right_eye_top":    386,
    "right_eye_bottom": 374,
    # 眉
    "left_eyebrow_inner":  107,
    "left_eyebrow_peak":   105,
    "left_eyebrow_outer":   46,
    "right_eyebrow_inner": 336,
    "right_eyebrow_peak":  334,
    "right_eyebrow_outer": 276,
    # 鼻
    "nose_top":   6,
    "nose_tip":   1,
    "nose_left":  129,
    "nose_right": 358,
    # 口
    "mouth_left":   61,
    "mouth_right":  291,
    "mouth_top":    13,
    "mouth_bottom": 14,
}

# 顎ライン（輪郭の丸み計算用）
JAW_LINE_INDICES = [
    234, 93, 132, 58, 172, 136, 150, 149, 176, 152,
    400, 378, 379, 365, 397, 288, 361, 323, 454,
]

# 分類閾値
MATURITY_THRESHOLD: float = 0.2
LINEARITY_THRESHOLD: float = 0.2


# タイプごとの表示情報
FACE_TYPE_INFO: dict[FaceType, dict] = {
    FaceType.CUTE: {
        "emoji": "🌸",
        "axis": "幼顔 × 曲線系",
        "description": "丸みのある目・ふっくらしたフェイスラインが特徴。愛らしく親しみやすい印象を与えます。",
        "keywords": ["かわいい", "親しみやすい", "やわらか"],
    },
    FaceType.ACTIVE_CUTE: {
        "emoji": "⚡",
        "axis": "幼顔 × 直線系",
        "description": "シャープな輪郭に幼さが融合。活発でスポーティな雰囲気を持ちます。",
        "keywords": ["フレッシュ", "スポーティ", "活発"],
    },
    FaceType.FRESH: {
        "emoji": "🌿",
        "axis": "幼顔 × 中間",
        "description": "清潔感のある爽やかな顔立ち。自然な若々しさが魅力です。",
        "keywords": ["爽やか", "清潔感", "自然"],
    },
    FaceType.FEMININE: {
        "emoji": "🌹",
        "axis": "大人顔 × 曲線系",
        "description": "柔らかな曲線と落ち着きのある雰囲気。女性らしい上品さが際立ちます。",
        "keywords": ["女性らしい", "やさしい", "上品"],
    },
    FaceType.COOL: {
        "emoji": "💎",
        "axis": "大人顔 × 直線系",
        "description": "シャープな輪郭と知的な雰囲気。クールで洗練された印象を与えます。",
        "keywords": ["クール", "知的", "洗練"],
    },
    FaceType.SOFT_ELEGANT: {
        "emoji": "🌙",
        "axis": "大人顔 × 中間",
        "description": "落ち着いた雰囲気の中に柔らかさを兼ね備えた顔立ち。エレガントな印象です。",
        "keywords": ["エレガント", "落ち着き", "上品"],
    },
    FaceType.COOL_CASUAL: {
        "emoji": "🎯",
        "axis": "中間 × 直線系",
        "description": "スタイリッシュでさっぱりとした印象。カジュアルな場でも存在感があります。",
        "keywords": ["スタイリッシュ", "さっぱり", "存在感"],
    },
    FaceType.ELEGANT: {
        "emoji": "✨",
        "axis": "中間 × 中間",
        "description": "幼さと大人っぽさ、柔らかさとシャープさのバランスが取れた顔立ち。どんな場にも馴染む万能タイプです。",
        "keywords": ["バランス型", "万能", "自然"],
    },
}
