"""顔分析アプリ — Streamlit エントリーポイント."""
from __future__ import annotations

import os
# DeepFace (0.0.93) は Keras 2 API を使用するため、TF 2.17+ で Keras 3 が
# デフォルトになる前に tf-keras (レガシー互換) を強制する。
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import io

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

from src.analyzer import FaceResult, analyze_image, init_insightface
from src.drawing import crop_face, draw_faces
from src.face_type import (
    FaceMetrics,
    FaceType,
    classify,
    compute_metrics,
    detect_landmarks,
    get_face_mesh,
)
from src.face_type.constants import FACE_TYPE_INFO
from src.i18n import gender_ja, race_ja
from src.similarity import FaceDetection, SimilarityResult, compute_similarity, detect_face

MAX_LONG_SIDE = 2000


st.set_page_config(page_title="顔分析アプリ", page_icon="🙂", layout="wide")


# ---------------------------------------------------------------------------
# モデルの warmup
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="モデルを読み込んでいます...")
def warmup_models() -> bool:
    dummy = np.zeros((224, 224, 3), dtype=np.uint8)
    try:
        analyze_image(dummy)
    except Exception:
        pass
    try:
        init_insightface()
    except Exception:
        pass
    return True


@st.cache_resource(show_spinner="顔タイプ診断モデルを準備しています...")
def get_cached_face_mesh():
    return get_face_mesh()


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


# ---------------------------------------------------------------------------
# タブ 1: 顔分析
# ---------------------------------------------------------------------------

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


def tab_analyze() -> None:
    st.subheader("画像から年齢・性別・人種カテゴリを推定")
    st.info(
        "⚠️ **ご利用にあたっての注意**\n\n"
        "- 本アプリの結果は機械学習モデルによる**統計的な推定値**です。"
        "個人の属性を断定するものではありません。\n"
        "- 「人種カテゴリ」は DeepFace が出力する分類ラベル "
        "(asian / white / black / middle eastern / indian / latino hispanic) に基づきます。\n"
        "- 差別的な目的での利用を固く禁じます。\n"
        "- アップロードされた画像はサーバに保存されません。"
    )
    uploaded = st.file_uploader(
        "顔が写っている画像をアップロードしてください(JPEG / PNG、最大 10MB)",
        type=["jpg", "jpeg", "png"],
        key="analyze_uploader",
    )
    if uploaded is None:
        return
    try:
        image_rgb = load_image(uploaded)
    except Exception as e:
        st.error(f"画像の読み込みに失敗しました: {e}")
        return
    with st.spinner("顔を解析しています..."):
        try:
            faces = analyze_image(image_rgb)
        except Exception as e:
            st.error(f"解析中にエラーが発生しました: {e}")
            return
    if not faces:
        st.warning("顔を検出できませんでした。別の画像をお試しください。")
        st.image(image_rgb, caption="アップロード画像", use_column_width=True)
        return
    annotated = draw_faces(image_rgb, faces)
    st.subheader(f"検出結果 — {len(faces)} 人の顔を検出")
    st.image(annotated, caption="検出結果", use_column_width=True)
    st.subheader("顔ごとの詳細")
    for idx, face in enumerate(faces, start=1):
        render_face_detail(idx, image_rgb, face)


# ---------------------------------------------------------------------------
# タブ 2: 顔の類似度判定
# ---------------------------------------------------------------------------

# ゲーミフィケーション: スコア帯ごとのレベル定義
_SIMILARITY_LEVELS = [
    (95, "🧬", "双子レベル！",         "もはや同一人物？！信じられないほどそっくりです！"),
    (85, "👯", "瓜二つレベル！",        "双子と間違えられること間違いなし。驚くほどよく似ています！"),
    (75, "👨‍👩‍👧", "親戚レベル",          "家族や親戚と言われても納得の似ている度合いです。"),
    (65, "😊", "似た者同士レベル",      "顔の特徴がよく合っています。仲良くなれそうな予感！"),
    (55, "🤔", "なんとなく似ているレベル", "よく見ると似ているところがあります。"),
    (45, "😐", "ちょっぴり似ているレベル", "ほんの少し共通点があります。"),
    (35, "🙂", "別人かもしれないレベル", "あまり似ていませんが、どこか引かれ合うものがあるかも？"),
    ( 0, "😅", "まったくの別人レベル",  "顔の特徴がかなり異なります。真逆のタイプかも！"),
]


def _get_similarity_level(score: float) -> tuple[str, str, str]:
    for threshold, emoji, name, desc in _SIMILARITY_LEVELS:
        if score >= threshold:
            return emoji, name, desc
    return "😅", "まったくの別人レベル", "顔の特徴がかなり異なります。"


def _score_color(score: float) -> str:
    if score >= 75:
        return "#22c55e"
    if score >= 50:
        return "#f59e0b"
    return "#ef4444"


def render_similarity_result(result: SimilarityResult, det_a: FaceDetection, det_b: FaceDetection) -> None:
    emoji, level_name, level_desc = _get_similarity_level(result.overall_score)
    color = _score_color(result.overall_score)

    # --- 顔並べ + 総合スコア ---
    col_face_a, col_vs, col_score, col_vs2, col_face_b = st.columns([2, 1, 3, 1, 2])
    with col_face_a:
        st.markdown("**画像 A**")
        if det_a.crop.size > 0:
            st.image(det_a.crop, use_column_width=True)
    with col_vs:
        st.markdown("<div style='text-align:center;font-size:36px;padding-top:40px'>VS</div>", unsafe_allow_html=True)
    with col_score:
        st.markdown(
            f"""
            <div style="text-align:center;padding:16px 0;">
                <div style="font-size:80px;font-weight:900;line-height:1;color:{color};">
                    {result.overall_score:.1f}<span style="font-size:40px">%</span>
                </div>
                <div style="font-size:32px;margin-top:4px;">{emoji}</div>
                <div style="font-size:20px;font-weight:700;color:{color};margin-top:4px;">{level_name}</div>
                <div style="font-size:13px;color:#666;margin-top:6px;">{level_desc}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_vs2:
        st.markdown("<div style='text-align:center;font-size:36px;padding-top:40px'>VS</div>", unsafe_allow_html=True)
    with col_face_b:
        st.markdown("**画像 B**")
        if det_b.crop.size > 0:
            st.image(det_b.crop, use_column_width=True)

    st.markdown("---")

    # --- スコア内訳 ---
    st.markdown("#### スコア内訳")
    m1, m2 = st.columns(2)
    with m1:
        st.metric("顔埋め込み類似度（ArcFace）— 重み 70%", f"{result.embedding_score:.1f}%")
    with m2:
        st.metric("幾何学的特徴の類似度 — 重み 30%", f"{result.geometric_score:.1f}%")

    # --- 顔パーツ別スコア棒グラフ ---
    st.markdown("#### 顔パーツ別 類似スコア")
    rows = [
        {"特徴": fs.name, "スコア(%)": round(fs.score, 1),
         "画像Aの値": round(fs.value1, 4), "画像Bの値": round(fs.value2, 4)}
        for fs in result.feature_scores
    ]
    df = pd.DataFrame(rows)
    st.bar_chart(df.set_index("特徴")["スコア(%)"], height=260)
    with st.expander("数値の詳細を見る"):
        st.dataframe(df, hide_index=True)


def tab_similarity() -> None:
    st.subheader("2 人の顔の似ている度合いを判定")
    st.info(
        "2 枚の画像をアップロードすると、それぞれに写っている人物の顔を比較して類似度を算出します。\n\n"
        "- **顔埋め込みベクトル**: ArcFace (InsightFace buffalo_l) の 512 次元特徴量 — 重み 70%\n"
        "- **幾何学的特徴**: 目の間隔・鼻の位置・口幅・顔の縦横比など 6 項目 — 重み 30%\n"
        "- アップロードされた画像はサーバに保存されません。"
    )
    col_a, col_b = st.columns(2)
    with col_a:
        file_a = st.file_uploader("画像 A（1 人目）", type=["jpg", "jpeg", "png"], key="sim_a")
    with col_b:
        file_b = st.file_uploader("画像 B（2 人目）", type=["jpg", "jpeg", "png"], key="sim_b")

    if file_a is None or file_b is None:
        return

    try:
        img_a = load_image(file_a)
        img_b = load_image(file_b)
    except Exception as e:
        st.error(f"画像の読み込みに失敗しました: {e}")
        return

    with st.spinner("顔を検出・解析しています..."):
        det_a = detect_face(img_a)
        det_b = detect_face(img_b)

    if det_a is None:
        st.error("画像 A から顔を検出できませんでした。別の画像をお試しください。")
        return
    if det_b is None:
        st.error("画像 B から顔を検出できませんでした。別の画像をお試しください。")
        return

    result = compute_similarity(det_a, det_b)
    render_similarity_result(result, det_a, det_b)


# ---------------------------------------------------------------------------
# タブ 3: 顔タイプ診断
# ---------------------------------------------------------------------------

def _draw_type_map(maturity: float, linearity: float, face_type: FaceType) -> plt.Figure:
    """2軸マップを描画して返す."""
    fig, ax = plt.subplots(figsize=(5, 5))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    # --- 4 象限の背景色 ---
    region_alpha = 0.12
    regions = [
        # (x_range, y_range, color, label)
        ((-1, 0), ( 0, 1), "#f472b6", "フェミニン / ソフトエレガント"),
        (( 0, 1), ( 0, 1), "#60a5fa", "クール / ソフトエレガント"),
        ((-1, 0), (-1, 0), "#4ade80", "キュート / フレッシュ"),
        (( 0, 1), (-1, 0), "#fbbf24", "アクティブキュート / フレッシュ"),
    ]
    for (x0, x1), (y0, y1), color, _ in regions:
        rect = mpatches.FancyBboxPatch(
            (x0, y0), x1 - x0, y1 - y0,
            boxstyle="square,pad=0",
            facecolor=color, alpha=region_alpha, edgecolor="none",
        )
        ax.add_patch(rect)

    # --- 軸線 ---
    ax.axhline(0, color="#555", linewidth=0.8, linestyle="--")
    ax.axvline(0, color="#555", linewidth=0.8, linestyle="--")

    # --- 軸ラベル ---
    label_kw = dict(color="#aaa", fontsize=9, ha="center", va="center")
    ax.text( 0.0,  1.05, "大人顔", **label_kw)
    ax.text( 0.0, -1.05, "幼顔",   **label_kw)
    ax.text( 1.08,  0.0, "直線系", **label_kw, ha="left")
    ax.text(-1.08,  0.0, "曲線系", **label_kw, ha="right")

    # --- プロット点 ---
    ax.scatter(
        [linearity], [maturity],
        s=200, zorder=5,
        color="#f97316", edgecolors="white", linewidths=1.5,
    )
    info = FACE_TYPE_INFO[face_type]
    ax.annotate(
        f"{info['emoji']} {face_type.value}",
        (linearity, maturity),
        xytext=(linearity + 0.05, maturity + 0.08),
        fontsize=9, color="white",
        arrowprops=dict(arrowstyle="->", color="#aaa", lw=0.8),
    )

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_xlabel("← 曲線系　　　　直線系 →", color="#aaa", fontsize=8)
    ax.set_ylabel("← 幼顔　　　　大人顔 →", color="#aaa", fontsize=8)
    ax.tick_params(colors="#555")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    ax.set_title("顔タイプ 2 軸マップ", color="#ccc", fontsize=10, pad=8)

    fig.tight_layout()
    return fig


def _render_metrics_detail(m: FaceMetrics) -> None:
    """各特徴量の値と解説をテーブル表示する."""
    rows = [
        ("顔の縦横比",    m.aspect_ratio,       "高い=面長、低い=丸顔"),
        ("目の縦位置",    m.eye_position_ratio,  "低い=目が顔の上寄り"),
        ("目幅比",        m.eye_width_ratio,     "大きい=大きな目"),
        ("鼻の長さ比",    m.nose_length_ratio,   "大きい=長い鼻"),
        ("下顔面比",      m.lower_face_ratio,    "大きい=鼻〜顎が長い"),
        ("顎の角度 (°)", m.jaw_angle,            "小さい=シャープ、大きい=丸み"),
        ("顔の丸み",      m.face_roundness,      "1.0 に近いほど丸顔"),
        ("目の丸み",      m.eye_roundness,       "大きい=丸い目"),
        ("眉の曲率",      m.eyebrow_curvature,   "大きい=アーチ型眉"),
    ]
    df = pd.DataFrame(rows, columns=["特徴", "値", "解説"])
    df["値"] = df["値"].apply(lambda v: round(float(v), 4))
    st.dataframe(df, hide_index=True)


def tab_face_type() -> None:
    st.subheader("あなたの顔タイプを診断します")
    st.info(
        "顔の各パーツを計測し、**大人っぽさ（成熟度）** と **直線的か曲線的か（直線性）** の "
        "2 軸でスコア化して 8 タイプに分類します。\n\n"
        "- 使用モデル: MediaPipe FaceLandmarker (478 点ランドマーク)\n"
        "- アップロードされた画像はサーバに保存されません。"
    )

    uploaded = st.file_uploader(
        "顔が写っている画像をアップロードしてください（JPEG / PNG、最大 10MB）",
        type=["jpg", "jpeg", "png"],
        key="facetype_uploader",
    )
    if uploaded is None:
        return

    try:
        image_rgb = load_image(uploaded)
    except Exception as e:
        st.error(f"画像の読み込みに失敗しました: {e}")
        return

    image_bgr = image_rgb[:, :, ::-1].copy()
    h_img, w_img = image_rgb.shape[:2]

    with st.spinner("ランドマークを検出しています..."):
        try:
            mesh = get_cached_face_mesh()
            landmarks = detect_landmarks(image_bgr, landmarker=mesh)
        except Exception as e:
            st.error(f"ランドマーク検出中にエラーが発生しました: {e}")
            return

    if landmarks is None:
        st.warning("顔のランドマークを検出できませんでした。正面を向いた顔写真をお試しください。")
        st.image(image_rgb, use_column_width=True)
        return

    metrics   = compute_metrics(landmarks, w_img, h_img)
    face_type, metrics = classify(metrics)
    info      = FACE_TYPE_INFO[face_type]

    # --- 結果表示 ---
    st.markdown("---")
    col_img, col_result = st.columns([1, 2])

    with col_img:
        st.image(image_rgb, caption="アップロード画像", use_column_width=True)

    with col_result:
        st.markdown(
            f"""
            <div style="padding:20px;border-radius:12px;background:#1a1f2e;text-align:center;">
                <div style="font-size:64px;line-height:1;">{info['emoji']}</div>
                <div style="font-size:28px;font-weight:900;color:#f97316;margin-top:8px;">
                    {face_type.value}
                </div>
                <div style="font-size:14px;color:#94a3b8;margin-top:4px;">{info['axis']}</div>
                <div style="font-size:15px;color:#e2e8f0;margin-top:12px;line-height:1.6;">
                    {info['description']}
                </div>
                <div style="margin-top:14px;">
                    {'　'.join(f'<span style="background:#374151;color:#d1d5db;padding:3px 10px;border-radius:999px;font-size:12px;">{kw}</span>' for kw in info['keywords'])}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        m_col, l_col = st.columns(2)
        with m_col:
            maturity_pct = (metrics.maturity_score + 1) / 2 * 100
            st.metric(
                "成熟度スコア",
                f"{maturity_pct:.0f}%",
                help="0% = 幼顔 / 100% = 大人顔",
            )
        with l_col:
            linearity_pct = (metrics.linearity_score + 1) / 2 * 100
            st.metric(
                "直線性スコア",
                f"{linearity_pct:.0f}%",
                help="0% = 曲線系 / 100% = 直線系",
            )

    # --- 2 軸マップ ---
    st.markdown("---")
    map_col, detail_col = st.columns([1, 1])
    with map_col:
        fig = _draw_type_map(metrics.maturity_score, metrics.linearity_score, face_type)
        st.pyplot(fig)
        plt.close(fig)

    with detail_col:
        st.markdown("#### 各特徴量の詳細")
        with st.expander("計測値を確認する", expanded=False):
            _render_metrics_detail(metrics)

        # 8 タイプ早見表
        st.markdown("#### 8 タイプ早見表")
        type_rows = [
            {
                "タイプ": f"{v['emoji']} {k.value}",
                "軸": v["axis"],
            }
            for k, v in FACE_TYPE_INFO.items()
        ]
        st.dataframe(pd.DataFrame(type_rows), hide_index=True)


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------

def main() -> None:
    st.title("🙂 顔分析アプリ")
    warmup_models()

    tab1, tab2, tab3 = st.tabs(["顔分析", "顔の類似度判定", "顔タイプ診断"])

    with tab1:
        tab_analyze()
    with tab2:
        tab_similarity()
    with tab3:
        tab_face_type()


if __name__ == "__main__":
    main()
