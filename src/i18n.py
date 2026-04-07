"""英語ラベル → 日本語ラベルのマッピング."""

GENDER_JA = {
    "Man": "男性",
    "Woman": "女性",
}

RACE_JA = {
    "asian": "アジア系",
    "white": "白人系",
    "black": "黒人系",
    "middle eastern": "中東系",
    "indian": "インド系",
    "latino hispanic": "ラテン系",
}


def gender_ja(label: str) -> str:
    return GENDER_JA.get(label, label)


def race_ja(label: str) -> str:
    return RACE_JA.get(label, label)
