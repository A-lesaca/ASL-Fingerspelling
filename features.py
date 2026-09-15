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

# ---------------------------------------------------------------------------
# Geometric feature extraction
# ---------------------------------------------------------------------------
# The normalised coordinates above are already invariant to where the hand is,
# how big it looks and how the wrist is tilted. What they do NOT make explicit
# is the thing that actually separates letters: which fingers are folded, how
# far apart the fingertips are, and how bent each joint is. A classifier can in
# principle infer all of that from raw coordinates, but stating it directly
# gives a small model far less to learn -- which matters when the dataset is a
# few thousand hand-labelled frames rather than a few million.

# Landmark chains, wrist-first, for each digit.
FINGER_CHAINS = {
    "thumb": (0, 1, 2, 3, 4),
    "index": (0, 5, 6, 7, 8),
    "middle": (0, 9, 10, 11, 12),
    "ring": (0, 13, 14, 15, 16),
    "pinky": (0, 17, 18, 19, 20),
}
FINGERTIPS = (4, 8, 12, 16, 20)
FINGER_MCPS = (1, 5, 9, 13, 17)

N_COORDS = FEATURE_DIM                                   # 63
N_DISTANCES = 10 + len(FINGERTIPS) + len(FINGERTIPS)     # 20
N_ANGLES = len(FINGER_CHAINS) * 3                        # 15
N_RATIOS = len(FINGER_CHAINS)                            # 5
VECTOR_DIM = N_COORDS + N_DISTANCES + N_ANGLES + N_RATIOS  # 103


def _pairwise_tip_distances(pts: np.ndarray) -> list[float]:
    """Distances between every pair of fingertips: the 'spread' of the hand."""
    out = []
    for i in range(len(FINGERTIPS)):
        for j in range(i + 1, len(FINGERTIPS)):
            out.append(float(np.linalg.norm(pts[FINGERTIPS[i]] - pts[FINGERTIPS[j]])))
    return out


def _joint_angles(pts: np.ndarray) -> list[float]:
    """Interior angle at each of the three joints along every finger.

    Returned in radians. A straight finger gives angles near pi; a fully
    curled one gives angles near zero. This is what distinguishes, say, a
    closed fist from a flat palm far more directly than coordinates do.
    """
    out = []
    for chain in FINGER_CHAINS.values():
        for a, b, c in zip(chain[:-2], chain[1:-1], chain[2:]):
            v1 = pts[a] - pts[b]
            v2 = pts[c] - pts[b]
            denom = np.linalg.norm(v1) * np.linalg.norm(v2)
            if denom < _EPS:
                out.append(0.0)
                continue
            cosine = float(np.dot(v1, v2) / denom)
            out.append(float(np.arccos(np.clip(cosine, -1.0, 1.0))))
    return out


def _extension_ratios(pts: np.ndarray) -> list[float]:
    """Straight-line tip-to-wrist distance over the finger's total bone length.

    Near 1.0 means the finger is extended; a curled finger travels the same
    bone length but ends up much closer to the wrist, so the ratio drops.
    """
    out = []
    for chain in FINGER_CHAINS.values():
        bone_length = sum(
            float(np.linalg.norm(pts[b] - pts[a]))
            for a, b in zip(chain[:-1], chain[1:])
        )
        if bone_length < _EPS:
            out.append(0.0)
            continue
        span = float(np.linalg.norm(pts[chain[-1]] - pts[chain[0]]))
        out.append(span / bone_length)
    return out


def build_features(landmarks: np.ndarray, mirror: bool = False) -> np.ndarray:
    """Full feature vector: coordinates, distances, angles, extension ratios.

    Everything is computed from the *normalised* landmarks, so the whole vector
    inherits their invariance to translation, scale and in-plane rotation.
    """
    coords = normalise_landmarks(landmarks, mirror=mirror)
    pts = coords.reshape(NUM_LANDMARKS, 3).astype(np.float64)

    distances = _pairwise_tip_distances(pts)
    distances += [float(np.linalg.norm(pts[t] - pts[WRIST])) for t in FINGERTIPS]
    distances += [
        float(np.linalg.norm(pts[t] - pts[m]))
        for t, m in zip(FINGERTIPS, FINGER_MCPS)
    ]

    vector = np.concatenate([
        coords,
        np.array(distances, dtype=np.float32),
        np.array(_joint_angles(pts), dtype=np.float32),
        np.array(_extension_ratios(pts), dtype=np.float32),
    ])
    if vector.shape != (VECTOR_DIM,):
        raise ValueError(f"expected {VECTOR_DIM} features, got {vector.shape}")
    return np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)