from __future__ import annotations

from typing import NamedTuple, Optional

import mediapipe as mp
import numpy as np

from .features import landmarks_to_array, normalise_landmarks, resolve_handedness


class Detection(NamedTuple):
    features: np.ndarray  # normalised landmark coordinates, shape (21, 3)
    handedness: str  # the user's true hand: "Left" or "Right"
    raw: object  # MediaPipe landmark list, kept for drawing


class HandDetector:
    """Thin wrapper over MediaPipe Hands returning ready-to-classify features."""

    def __init__(
        self,
        static_image_mode: bool = False,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.5,
        model_complexity: int = 1,
    ) -> None:
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=1,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def detect(self, rgb_frame: np.ndarray, frame_is_mirrored: bool) -> Optional[Detection]:
        """Detect one hand and return its normalised features, or None.

        Args:
            rgb_frame: HxWx3 RGB image (OpenCV gives BGR -- convert first).
            frame_is_mirrored: True if the frame was flipped for selfie view.
        """
        result = self._hands.process(rgb_frame)
        if not result.multi_hand_landmarks:
            return None

        raw = result.multi_hand_landmarks[0]
        label = result.multi_handedness[0].classification[0].label
        hand = resolve_handedness(label, frame_is_mirrored)

        features = normalise_landmarks(landmarks_to_array(raw), mirror=(hand == "Left"))
        return Detection(features=features, handedness=hand, raw=raw)

    def close(self) -> None:
        self._hands.close()

    def __enter__(self) -> "HandDetector":
        return self

    def __exit__(self, *exc) -> None:
        self.close()