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