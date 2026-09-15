"""Tests for landmark normalisation."""

import numpy as np
import pytest

from features import (
    FEATURE_DIM,
    MIDDLE_MCP,
    WRIST,
    normalise_landmarks,
    resolve_handedness,
)

rng = np.random.default_rng(0)


def fake_hand() -> np.ndarray:
    """A plausible random hand: 21 points in the middle of the frame."""
    return rng.uniform(0.3, 0.7, size=(21, 3))


def rotate_xy(pts: np.ndarray, theta: float, about: np.ndarray) -> np.ndarray:
    out = pts.copy()
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s], [s, c]])
    out[:, :2] = (out[:, :2] - about) @ rot.T + about
    return out


def test_output_shape_and_dtype():
    out = normalise_landmarks(fake_hand())
    assert out.shape == (FEATURE_DIM,)
    assert out.dtype == np.float32


def test_rejects_wrong_shape():
    with pytest.raises(ValueError):
        normalise_landmarks(np.zeros((20, 3)))
    with pytest.raises(ValueError):
        normalise_landmarks(np.zeros((21, 2)))


def test_wrist_maps_to_origin():
    out = normalise_landmarks(fake_hand()).reshape(21, 3)
    assert np.allclose(out[WRIST], [0.0, 0.0, 0.0], atol=1e-5)


def test_palm_bone_points_up_with_unit_length():
    """After normalising, landmark 9 must sit at (0, -1): the canonical pose."""
    out = normalise_landmarks(fake_hand()).reshape(21, 3)
    assert np.allclose(out[MIDDLE_MCP, :2], [0.0, -1.0], atol=1e-5)


def test_invariant_to_translation():
    hand = fake_hand()
    moved = hand + np.array([0.15, -0.08, 0.0])
    assert np.allclose(
        normalise_landmarks(hand), normalise_landmarks(moved), atol=1e-5
    )


def test_invariant_to_scale():
    """Hand held closer to the camera -> same letter, same features."""
    hand = fake_hand()
    bigger = (hand - hand[WRIST]) * 2.5 + hand[WRIST]
    assert np.allclose(
        normalise_landmarks(hand), normalise_landmarks(bigger), atol=1e-5
    )


@pytest.mark.parametrize("degrees", [15, 90, 180, -47])
def test_invariant_to_in_plane_rotation(degrees):
    """Tilting the wrist must not change the predicted letter's features."""
    hand = fake_hand()
    turned = rotate_xy(hand, np.deg2rad(degrees), about=hand[WRIST, :2])
    assert np.allclose(
        normalise_landmarks(hand), normalise_landmarks(turned), atol=1e-5
    )


def test_mirroring_changes_the_features():
    """Sanity check that the mirror flag is actually wired up."""
    hand = fake_hand()
    assert not np.allclose(
        normalise_landmarks(hand, mirror=False),
        normalise_landmarks(hand, mirror=True),
        atol=1e-3,
    )


def test_mirrored_left_hand_matches_the_right_hand_it_reflects():
    """A left hand, mirrored, should normalise identically to its reflection."""
    right = fake_hand()
    left = right.copy()
    left[:, 0] = 1.0 - left[:, 0]  
    assert np.allclose(
        normalise_landmarks(right, mirror=False),
        normalise_landmarks(left, mirror=True),
        atol=1e-5,
    )


def test_degenerate_hand_returns_zeros_not_nan():
    """Guard against inf/nan poisoning the model on a bad detection."""
    collapsed = np.full((21, 3), 0.5)
    out = normalise_landmarks(collapsed)
    assert np.all(np.isfinite(out))
    assert np.allclose(out, 0.0)


class TestResolveHandedness:
    def test_unmirrored_frame_passes_label_through(self):
        assert resolve_handedness("Left", frame_is_mirrored=False) == "Left"
        assert resolve_handedness("Right", frame_is_mirrored=False) == "Right"

    def test_mirrored_frame_inverts_label(self):
        assert resolve_handedness("Left", frame_is_mirrored=True) == "Right"
        assert resolve_handedness("Right", frame_is_mirrored=True) == "Left"

    def test_rejects_junk(self):
        with pytest.raises(ValueError):
            resolve_handedness("left", frame_is_mirrored=True)

class TestGeometricFeatures:
    """The coordinates/distances/angles/ratios vector built on top of them."""

    def test_vector_has_the_advertised_width(self):
        from features import (N_ANGLES, N_COORDS, N_DISTANCES, N_RATIOS,
                              VECTOR_DIM, build_features)
        out = build_features(fake_hand())
        assert out.shape == (VECTOR_DIM,)
        assert N_COORDS + N_DISTANCES + N_ANGLES + N_RATIOS == VECTOR_DIM
        assert out.dtype == np.float32

    def test_first_block_is_the_normalised_coordinates(self):
        from features import N_COORDS, build_features
        hand = fake_hand()
        assert np.allclose(
            build_features(hand)[:N_COORDS], normalise_landmarks(hand), atol=1e-6
        )

    @pytest.mark.parametrize("degrees", [30, 90, -120])
    def test_inherits_rotation_invariance(self, degrees):
        from features import build_features
        hand = fake_hand()
        turned = rotate_xy(hand, np.deg2rad(degrees), about=hand[WRIST, :2])
        assert np.allclose(build_features(hand), build_features(turned), atol=1e-4)

    def test_inherits_scale_invariance(self):
        from features import build_features
        hand = fake_hand()
        bigger = (hand - hand[WRIST]) * 3.0 + hand[WRIST]
        assert np.allclose(build_features(hand), build_features(bigger), atol=1e-4)

    def test_extension_ratio_separates_straight_from_curled(self):
        """A straight finger scores near 1.0; a curled one scores well below."""
        from features import N_ANGLES, N_RATIOS, build_features

        straight = np.zeros((21, 3))
        straight[9] = [0.0, -1.0, 0.0]          # palm bone, sets the scale
        for i, lm in enumerate((5, 6, 7, 8)):   # index finger, fully extended
            straight[lm] = [0.3, -0.4 * (i + 1), 0.0]

        curled = straight.copy()
        curled[7] = [0.3, -0.9, 0.0]            # fold the last two joints back
        curled[8] = [0.3, -0.5, 0.0]

        ratios = slice(-N_RATIOS, None)
        index_ratio = 1  # thumb, index, middle, ring, pinky
        s = build_features(straight)[ratios][index_ratio]
        c = build_features(curled)[ratios][index_ratio]
        assert s > 0.95, f"extended finger should score near 1.0, got {s}"
        assert c < s, f"curled finger should score lower than extended ({c} vs {s})"

    def test_angles_are_radians_in_range(self):
        from features import N_ANGLES, N_RATIOS, build_features
        angles = build_features(fake_hand())[-(N_ANGLES + N_RATIOS):-N_RATIOS]
        assert angles.shape == (N_ANGLES,)
        assert np.all(angles >= 0.0) and np.all(angles <= np.pi + 1e-6)

    def test_degenerate_hand_is_finite(self):
        from features import build_features
        out = build_features(np.full((21, 3), 0.5))
        assert np.all(np.isfinite(out))
