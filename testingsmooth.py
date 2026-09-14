"""Tests for the debouncer."""

import pytest

from aslfs.smoothing import DELETE, NO_HAND, SPACE, LetterDebouncer


def feed(deb: LetterDebouncer, label: str, n: int, conf: float = 1.0):
    """Push the same prediction n times, returning everything committed."""
    return [c for _ in range(n) if (c := deb.update(label, conf)) is not None]


@pytest.fixture
def deb():
    return LetterDebouncer(
        window=10, min_votes=7, min_confidence=0.8, release_votes=4
    )


class TestConstructor:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"window": 5, "min_votes": 6},
            {"min_votes": 0},
            {"min_confidence": 1.5},
            {"release_votes": 0},
        ],
    )
    def test_rejects_invalid_config(self, kwargs):
        with pytest.raises(ValueError):
            LetterDebouncer(**kwargs)


class TestVoting:
    def test_single_frame_does_not_commit(self, deb):
        assert deb.update("A") is None
        assert deb.text == ""

    def test_commits_once_votes_reached(self, deb):
        assert feed(deb, "A", 6) == []
        assert deb.update("A") == "A"
        assert deb.text == "A"

    def test_isolated_noise_is_ignored(self, deb):
        """One bad frame in the middle of a held sign must not leak through."""
        feed(deb, "A", 4)
        deb.update("Q")  # misfire
        feed(deb, "A", 4)
        assert deb.text == "A"

    def test_low_confidence_frames_do_not_vote(self, deb):
        assert feed(deb, "B", 20, conf=0.5) == []
        assert deb.text == ""

    def test_no_hand_never_commits(self, deb):
        assert feed(deb, NO_HAND, 20) == []
        assert deb.text == ""


class TestLatching:
    def test_held_sign_commits_only_once(self, deb):
        committed = feed(deb, "C", 60)
        assert committed == ["C"]
        assert deb.text == "C"

    def test_double_letter_requires_a_release(self, deb):
        """Signing L, dropping the hand, signing L again types 'LL'."""
        feed(deb, "L", 10)
        feed(deb, NO_HAND, 10)
        feed(deb, "L", 10)
        assert deb.text == "LL"

    def test_different_letter_commits_without_release(self, deb):
        feed(deb, "H", 10)
        feed(deb, "I", 10)
        assert deb.text == "HI"

    def test_latched_letter_is_reported(self, deb):
        feed(deb, "D", 10)
        assert deb.latched == "D"
        feed(deb, NO_HAND, 10)
        assert deb.latched is None


class TestControlLabels:
    def test_space_inserts_a_space(self, deb):
        feed(deb, "H", 10)
        feed(deb, NO_HAND, 6)
        feed(deb, SPACE, 10)
        assert deb.text == "H "

    def test_delete_removes_last_character(self, deb):
        feed(deb, "H", 10)
        feed(deb, "I", 10)
        feed(deb, NO_HAND, 6)
        feed(deb, DELETE, 10)
        assert deb.text == "H"

    def test_delete_on_empty_buffer_is_safe(self, deb):
        feed(deb, DELETE, 10)
        assert deb.text == ""


class TestProgress:
    def test_progress_is_zero_when_idle(self, deb):
        assert deb.progress() == (None, 0.0)

    def test_progress_rises_towards_commit(self, deb):
        feed(deb, "E", 3)
        label, frac = deb.progress()
        assert label == "E"
        assert 0.0 < frac < 1.0


def test_reset_clears_everything(deb):
    feed(deb, "A", 10)
    deb.reset()
    assert deb.text == ""
    assert deb.latched is None
    assert deb.progress() == (None, 0.0)