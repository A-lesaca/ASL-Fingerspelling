"""MediaPipe hand detection, configured in exactly one place.

Uses the MediaPipe **Tasks** API. The older `mp.solutions.hands` interface that
most online tutorials still show was deprecated in 2023 and has since been
removed from the package entirely -- on current versions it raises
``AttributeError: module 'mediapipe' has no attribute 'solutions'``.

Every script imports this rather than configuring MediaPipe itself. The reason
is train/serve skew: if the data-collection script and the live demo disagree
about detection confidence or mirroring, the model is evaluated on a subtly
different distribution from the one it meets at inference, and the failure is
silent.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import NamedTuple, Optional

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

from features import landmarks_to_array, normalise_landmarks, resolve_handedness

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_FILE = Path("hand_landmarker.task")

# (start, end) index pairs for drawing the hand skeleton.
CONNECTIONS = [
    (c.start, c.end) for c in vision.HandLandmarksConnections.HAND_CONNECTIONS
]


class Detection(NamedTuple):
    features: np.ndarray  # (63,) normalised, canonicalised to a right hand
    handedness: str  # the user's true hand: "Left" or "Right"
    raw: list  # the 21 NormalizedLandmarks, kept for drawing


def ensure_model(path: Path = MODEL_FILE) -> Path:
    """Download the hand landmarker model bundle if it isn't here yet (~7MB)."""
    if not path.exists():
        print(f"downloading hand landmarker model -> {path}")
        urllib.request.urlretrieve(MODEL_URL, path)
    return path


def draw_landmarks(bgr_frame: np.ndarray, landmarks: list) -> None:
    """Draw the hand skeleton onto a BGR frame, in place.

    The Tasks API dropped the old `drawing_utils` helper, so this replaces it.
    Landmark coordinates are image-normalised, hence the multiply by w and h.
    """
    h, w = bgr_frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in CONNECTIONS:
        cv2.line(bgr_frame, pts[a], pts[b], (255, 255, 255), 2)
    for p in pts:
        cv2.circle(bgr_frame, p, 4, (0, 160, 255), -1)


class HandDetector:
    """Wrapper over MediaPipe Hand Landmarker returning ready-to-classify features."""

    def __init__(
        self,
        static_image_mode: bool = False,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.5,
        model_path: Optional[Path] = None,
    ) -> None:
        model = ensure_model(Path(model_path) if model_path else MODEL_FILE)

        # IMAGE mode for unrelated stills, VIDEO mode for a webcam stream --
        # VIDEO enables frame-to-frame tracking, which is both faster and
        # steadier than re-detecting from scratch on every frame.
        self._mode = (
            vision.RunningMode.IMAGE if static_image_mode else vision.RunningMode.VIDEO
        )
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model)),
            running_mode=self._mode,
            num_hands=1,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._timestamp = 0

    def detect(
        self, rgb_frame: np.ndarray, frame_is_mirrored: bool
    ) -> Optional[Detection]:
        """Detect one hand and return its normalised features, or None.

        Args:
            rgb_frame: HxWx3 RGB image (OpenCV gives BGR -- convert first).
            frame_is_mirrored: True if the frame was flipped for selfie view.
        """
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        if self._mode == vision.RunningMode.VIDEO:
            # VIDEO mode requires strictly increasing timestamps.
            self._timestamp += 33  # ~30fps, in milliseconds
            result = self._landmarker.detect_for_video(image, self._timestamp)
        else:
            result = self._landmarker.detect(image)

        if not result.hand_landmarks:
            return None

        raw = result.hand_landmarks[0]
        label = result.handedness[0][0].category_name
        hand = resolve_handedness(label, frame_is_mirrored)

        features = normalise_landmarks(landmarks_to_array(raw), mirror=(hand == "Left"))
        return Detection(features=features, handedness=hand, raw=raw)

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "HandDetector":
        return self

    def __exit__(self, *exc) -> None:
        self.close()