"""顔タイプ定数."""
from __future__ import annotations

from enum import Enum


class FaceType(Enum):
    CUTE         = "キュート"
    ACTIVE_CUTE  = "アクティブキュート"
    FRESH        = "フレッシュ"
    FEMININE     = "フェミニン"
    COOL         = "クール"
    SOFT_ELEGANT = "ソフトエレガント"
    COOL_CASUAL  = "クールカジュアル"
    ELEGANT      = "エレガント"


# 大人/幼の分類閾値
MATURITY_THRESHOLD: float = 0.15
# 直線/曲線の分類閾値
LINEARITY_THRESHOLD: float = 0.15


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
