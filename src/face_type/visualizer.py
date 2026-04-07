"""結果の可視化: ランドマーク描画・レーダーチャート・ポジションチャート."""
import cv2
import numpy as np
import plotly.graph_objects as go
from mediapipe.tasks.python.vision import FaceLandmarksConnections

from .constants import FaceType, LANDMARKS
from .metrics import FaceMetrics


def _connection_pairs(connection_set) -> list[tuple[int, int]]:
    pairs = []
    for c in connection_set:
        start = getattr(c, "start", None)
        end   = getattr(c, "end",   None)
        if start is None:
            start, end = c[0], c[1]
        pairs.append((start, end))
    return pairs


_TESSELATION = _connection_pairs(FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION)
_CONTOURS = (
    _connection_pairs(FaceLandmarksConnections.FACE_LANDMARKS_FACE_OVAL)
    + _connection_pairs(FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYE)
    + _connection_pairs(FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYE)
    + _connection_pairs(FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYEBROW)
    + _connection_pairs(FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYEBROW)
    + _connection_pairs(FaceLandmarksConnections.FACE_LANDMARKS_LIPS)
    + _connection_pairs(FaceLandmarksConnections.FACE_LANDMARKS_NOSE)
)


def draw_landmarks_on_image(
    image: np.ndarray, landmarks, mode: str = "mesh"
) -> np.ndarray:
    """landmarks を RGB 画像に重ね描きして返す.

    mode: "mesh" | "points" | "none"
    """
    out = image.copy()
    if mode == "none" or landmarks is None:
        return out

    h, w = out.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    n   = len(pts)

    if mode == "mesh":
        for s, e in _TESSELATION:
            if s < n and e < n:
                cv2.line(out, pts[s], pts[e], (180, 180, 180), 1, cv2.LINE_AA)
        for s, e in _CONTOURS:
            if s < n and e < n:
                cv2.line(out, pts[s], pts[e], (0, 220, 100), 1, cv2.LINE_AA)
    elif mode == "points":
        for _, idx in LANDMARKS.items():
            if idx < n:
                cv2.circle(out, pts[idx], 3, (0, 255, 0), -1)
    return out


def create_radar_chart(metrics: FaceMetrics) -> go.Figure:
    """特徴量レーダーチャートを描画する.

    各軸のスケールは日本人統計基準:
    - 縦横比   : 1.25〜1.55 を 0〜1 にマップ (日本人平均 ~1.40)
    - 目の大きさ: 0.15〜0.27 を 0〜1 にマップ (日本人平均 ~0.20)
    - 目の丸み  : 0.14〜0.38 を 0〜1 にマップ (日本人平均 ~0.24)
    - 顎の鋭さ  : 112〜132° を逆スケール    (日本人平均 ~120°)
    - 輪郭の丸み: 0.60〜0.90 を 0〜1 にマップ (日本人平均 ~0.76)
    - 眉の曲率  : 0.00〜0.022を 0〜1 にマップ
    """
    categories = ["縦横比\n(面長↑)", "目の大きさ", "目の丸み", "顎の鋭さ\n(シャープ↑)", "輪郭の丸み", "眉の曲率\n(アーチ↑)"]
    values = [
        min(max((metrics.aspect_ratio    - 1.25) / 0.30, 0.0), 1.0),
        min(max((metrics.eye_width_ratio - 0.15) / 0.12, 0.0), 1.0),
        min(max((metrics.eye_roundness   - 0.14) / 0.24, 0.0), 1.0),
        min(max((132 - metrics.jaw_angle)         / 20,  0.0), 1.0),  # 逆スケール: 小=シャープ
        min(max((metrics.face_roundness  - 0.60) / 0.30, 0.0), 1.0),
        min(max(metrics.eyebrow_curvature          / 0.022, 0.0), 1.0),
    ]
    values.append(values[0])
    categories.append(categories[0])

    fig = go.Figure(
        data=go.Scatterpolar(
            r=values, theta=categories, fill="toself",
            line=dict(color="#f97316"), fillcolor="rgba(249,115,22,0.25)",
        )
    )
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(size=9)),
            angularaxis=dict(tickfont=dict(size=11)),
        ),
        showlegend=False,
        title=dict(text="特徴量レーダーチャート", font=dict(size=13)),
        height=370,
        margin=dict(t=50, b=20, l=40, r=40),
    )
    return fig


_TYPE_POSITIONS: dict[FaceType, tuple[float, float]] = {
    FaceType.CUTE:         (-0.60, -0.60),
    FaceType.FRESH:        ( 0.00, -0.60),
    FaceType.ACTIVE_CUTE:  ( 0.60, -0.60),
    FaceType.FEMININE:     (-0.60,  0.60),
    FaceType.SOFT_ELEGANT: (-0.30,  0.30),
    FaceType.ELEGANT:      ( 0.00,  0.00),
    FaceType.COOL_CASUAL:  ( 0.30,  0.00),
    FaceType.COOL:         ( 0.60,  0.60),
}


def create_position_chart(
    maturity: float, linearity: float, face_type: FaceType
) -> go.Figure:
    from .constants import FACE_TYPE_INFO

    fig = go.Figure()

    # 背景ゾーン
    fig.add_shape(type="rect", x0=-1, x1=0, y0=-1,  y1=0,  fillcolor="rgba(74,222,128,0.08)",  line_width=0)
    fig.add_shape(type="rect", x0= 0, x1=1, y0=-1,  y1=0,  fillcolor="rgba(251,191,36,0.08)",  line_width=0)
    fig.add_shape(type="rect", x0=-1, x1=0, y0= 0,  y1=1,  fillcolor="rgba(244,114,182,0.08)", line_width=0)
    fig.add_shape(type="rect", x0= 0, x1=1, y0= 0,  y1=1,  fillcolor="rgba(96,165,250,0.08)",  line_width=0)

    # 全タイプをグレー点でプロット
    xs, ys, texts = [], [], []
    for t, (x, y) in _TYPE_POSITIONS.items():
        info = FACE_TYPE_INFO[t]
        xs.append(x)
        ys.append(y)
        texts.append(f"{info['emoji']} {t.value}")
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers+text", text=texts, textposition="top center",
        marker=dict(size=10, color="rgba(200,200,200,0.6)"),
        textfont=dict(size=10, color="#aaa"),
        name="タイプ",
    ))

    # 診断結果を星マークで強調
    info = FACE_TYPE_INFO[face_type]
    fig.add_trace(go.Scatter(
        x=[linearity], y=[maturity], mode="markers+text",
        text=[f"{info['emoji']} {face_type.value}"],
        textposition="bottom center",
        marker=dict(size=22, color="#f97316", symbol="star", line=dict(color="white", width=1.5)),
        textfont=dict(size=13, color="#f97316"),
        name="あなた",
    ))

    fig.update_layout(
        title=dict(text="8タイプ ポジションマップ", font=dict(size=13)),
        xaxis=dict(title="← 曲線系　　直線系 →", range=[-1.1, 1.1], zeroline=True,
                   zerolinecolor="#555", gridcolor="#222"),
        yaxis=dict(title="← 幼顔　　大人顔 →",   range=[-1.1, 1.1], zeroline=True,
                   zerolinecolor="#555", gridcolor="#222"),
        height=400,
        showlegend=False,
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font=dict(color="#ccc"),
        margin=dict(t=50, b=50, l=60, r=20),
    )
    return fig
