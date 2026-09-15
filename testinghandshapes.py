"""Tests for the handshape guidance and thumbnail naming."""

import string

import pytest

from app import BUILT_IN, Engine
from handshapes import HANDSHAPES, describe

STATIC_LETTERS = {c for c in string.ascii_uppercase if c not in {"J", "Z"}}


def test_every_teachable_built_in_has_guidance():
    assert set(BUILT_IN) <= set(HANDSHAPES)


def test_motion_letters_are_not_described():
    """J and Z are excluded from the model, so describing them would mislead."""
    assert "J" not in HANDSHAPES and "Z" not in HANDSHAPES


def test_all_static_letters_are_covered():
    assert STATIC_LETTERS <= set(HANDSHAPES)


@pytest.mark.parametrize("letter", sorted(STATIC_LETTERS))
def test_descriptions_are_usable_sentences(letter):
    text = HANDSHAPES[letter]
    assert len(text) > 15
    assert text[0].isupper() and text.endswith(".")


def test_custom_gesture_gets_a_fallback_description():
    assert "HELLO" in describe("HELLO")


class TestThumbNames:
    @pytest.mark.parametrize("label,expected", [
        ("A", "A"),
        ("space", "space"),
        ("THANK YOU", "THANK_YOU"),
        ("IT'S", "IT_S"),
        ("../escape", "___escape"),
    ])
    def test_names_are_filename_safe(self, label, expected):
        assert Engine.thumb_name(label) == expected

    def test_path_traversal_cannot_escape_the_folder(self):
        """A gesture name must never be able to point outside data/thumbs."""
        name = Engine.thumb_name("../../etc/passwd")
        assert "/" not in name and ".." not in name