"""MediaPipe Tasks API (FaceLandmarker) によるランドマーク検出."""
import urllib.request
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
_MODEL_DIR = Path(__file__).parent.parent.parent / "assets" / "models"
_MODEL_PATH = _MODEL_DIR / "face_landmarker.task"


def _ensure_model() -> Path:
    if not _MODEL_PATH.exists():
        _MODEL_DIR.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
    return _MODEL_PATH


class _FaceMeshWrapper:
    """BGR 画像を受け取って detect する薄いラッパー."""

    def __init__(self, landmarker):
        self._lm = landmarker

    def detect(self, image_bgr: np.ndarray):
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        return self._lm.detect(mp_image)

    def close(self):
        self._lm.close()


def get_face_mesh() -> _FaceMeshWrapper:
    model_path = _ensure_model()
    options = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return _FaceMeshWrapper(mp_vision.FaceLandmarker.create_from_options(options))


def detect_landmarks(image_bgr: np.ndarray, landmarker=None) -> Optional[list]:
    owns = landmarker is None
    if owns:
        landmarker = get_face_mesh()
    try:
        result = landmarker.detect(image_bgr)
    finally:
        if owns:
            landmarker.close()
    if not result.face_landmarks:
        return None
    return result.face_landmarks[0]
