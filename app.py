"""顔分析アプリ — Streamlit エントリーポイント."""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

from src.analyzer import FaceResult, analyze_image
from src.drawing import crop_face, draw_faces
from src.i18n import gender_ja, race_ja

MAX_LONG_SIDE = 2000


st.set_page_config(page_title="顔分析アプリ", page_icon="🙂", layout="wide")


@st.cache_resource(show_spinner="モデルを読み込んでいます...")
def warmup_models() -> bool:
    """初回起動時に DeepFace のモデルをロード(ダミー画像を1回 analyze)."""
    dummy = np.zeros((224, 224, 3), dtype=np.uint8)
    try:
        analyze_image(dummy)
    except Exception:
        # 顔が検出されないのは想定内
        pass
    return True


def load_image(uploaded_file) -> np.ndarray:
    """アップロードされたファイルを RGB ndarray に変換し、必要に応じてリサイズ."""
    image = Image.open(io.BytesIO(uploaded_file.read()))
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")

    w, h = image.size
    long_side = max(w, h)
    if long_side > MAX_LONG_SIDE:
        scale = MAX_LONG_SIDE / long_side
        image = image.resize((int(w * scale), int(h * scale)))

    return np.array(image)


def render_face_detail(idx: int, image_rgb: np.ndarray, face: FaceResult) -> None:
    with st.container(border=True):
        cols = st.columns([1, 2])
        with cols[0]:
            thumb = crop_face(image_rgb, face)
            if thumb.size > 0:
                st.image(thumb, caption=f"#{idx}", use_column_width=True)
        with cols[1]:
            st.markdown(f"### 顔 #{idx}")
            st.markdown(f"- **推定年齢**: {face.age} 歳")
            st.markdown(
                f"- **性別**: {gender_ja(face.gender)} "
                f"({face.gender}) — 確率 {face.gender_confidence:.1f}%"
            )
            st.markdown(
                f"- **人種カテゴリ(最有力)**: {race_ja(face.dominant_race)} "
                f"({face.dominant_race})"
            )

            if face.race:
                df = pd.DataFrame(
                    [
                        {"カテゴリ": race_ja(k), "ラベル": k, "確率(%)": round(v, 2)}
                        for k, v in face.race.items()
                    ]
                ).sort_values("確率(%)", ascending=False)
                st.markdown("**人種カテゴリの確率分布**")
                st.dataframe(df, hide_index=True)
                st.bar_chart(df.set_index("カテゴリ")["確率(%)"])


def main() -> None:
    st.title("🙂 顔分析アプリ")
    st.caption("画像から年齢・性別・人種カテゴリを統計的に推定します(DeepFace 使用)")

    st.info(
        "⚠️ **ご利用にあたっての注意**\n\n"
        "- 本アプリの結果は機械学習モデルによる**統計的な推定値**です。"
        "個人の属性を断定するものではありません。\n"
        "- 「人種カテゴリ」は DeepFace が出力する分類ラベル "
        "(asian / white / black / middle eastern / indian / latino hispanic) に基づきます。\n"
        "- 差別的な目的での利用を固く禁じます。\n"
        "- アップロードされた画像はサーバに保存されません。"
    )

    warmup_models()

    uploaded = st.file_uploader(
        "顔が写っている画像をアップロードしてください(JPEG / PNG、最大 10MB)",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded is None:
        st.stop()

    try:
        image_rgb = load_image(uploaded)
    except Exception as e:
        st.error(f"画像の読み込みに失敗しました: {e}")
        st.stop()

    with st.spinner("顔を解析しています..."):
        try:
            faces = analyze_image(image_rgb)
        except Exception as e:
            st.error(f"解析中にエラーが発生しました: {e}")
            st.stop()

    if not faces:
        st.warning("顔を検出できませんでした。別の画像をお試しください。")
        st.image(image_rgb, caption="アップロード画像", use_column_width=True)
        st.stop()

    annotated = draw_faces(image_rgb, faces)

    st.subheader(f"検出結果 — {len(faces)} 人の顔を検出")
    st.image(annotated, caption="検出結果", use_column_width=True)

    st.subheader("顔ごとの詳細")
    for idx, face in enumerate(faces, start=1):
        render_face_detail(idx, image_rgb, face)


if __name__ == "__main__":
    main()
