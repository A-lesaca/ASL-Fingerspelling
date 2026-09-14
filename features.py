from __future__ import annotations

import numpy as np

# Landmark indices, per the MediaPipe hand model.
WRIST = 0
MIDDLE_MCP = 9  # knuckle of the middle finger

NUM_LANDMARKS = 21
FEATURE_DIM = NUM_LANDMARKS * 3  # 63

_EPS = 1e-8


def normalise_landmarks(landmarks: np.ndarray, mirror: bool = False) -> np.ndarray:
    """Normalise a set of hand landmarks to a canonical pose."""
    pts = np.asarray(landmarks, dtype=np.float64)
    if pts.shape != (NUM_LANDMARKS, 3):
        raise ValueError(f"expected landmarks of shape (21, 3), got {pts.shape}")

    pts = pts.copy()

    # 1. Mirror the x-coordinates if requested.
    if mirror:
        pts[:, 0] = 1.0 - pts[:, 0]

    # 2. Translate the wrist to the origin.
    pts -= pts[WRIST]

    # 3. Scale by palm-bone length.
    scale = np.linalg.norm(pts[MIDDLE_MCP, :2])
    if scale < _EPS:
        return np.zeros(FEATURE_DIM, dtype=np.float32)
    pts /= scale

    # 4. Rotate about the origin so the palm bone points up.
    v = pts[MIDDLE_MCP, :2]
    angle = np.arctan2(v[1], v[0])
    theta = (-np.pi / 2.0) - angle
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    pts[:, :2] = pts[:, :2] @ rot.T

    return pts.reshape(-1).astype(np.float32)


def resolve_handedness(mediapipe_label: str, frame_is_mirrored: bool) -> str:
    """Return the user's true handedness."""
    if mediapipe_label not in ("Left", "Right"):
        raise ValueError(f"unexpected handedness label: {mediapipe_label!r}")
    if not frame_is_mirrored:
        return mediapipe_label
    return "Right" if mediapipe_label == "Left" else "Left"


def landmarks_to_array(hand_landmarks) -> np.ndarray:
    """Convert MediaPipe hand landmarks into a plain (21, 3) array.

    The Tasks API hands back a plain list of NormalizedLandmark. The older
    solutions API wrapped that list in a NormalizedLandmarkList with a
    ``.landmark`` attribute, so both shapes are accepted here.
    """
    points = getattr(hand_landmarks, "landmark", hand_landmarks)
    arr = np.array([[lm.x, lm.y, lm.z] for lm in points], dtype=np.float64)
    if arr.shape != (NUM_LANDMARKS, 3):
        raise ValueError(f"expected {NUM_LANDMARKS} landmarks, got {arr.shape}")
    return arr