"""Tests for the debouncer."""

import pytest

from smoothing import DELETE, NO_HAND, SPACE, LetterDebouncer


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

class TestReleaseDoesNotDoubleType:
    """Regression: a short release must not re-commit from stale frames.

    When the latch is released, the window can still be full of the sign the
    hand just left. Before the fix those frames immediately won the vote
    again, so signing G, dropping the hand and signing O produced 'GGOO'.
    """

    @pytest.mark.parametrize("release_votes", [3, 4, 5, 8])
    def test_alternating_letters_type_once_each(self, release_votes):
        d = LetterDebouncer(window=10, min_votes=5, min_confidence=0.5,
                            release_votes=release_votes)
        for _ in range(3):
            feed(d, "G", 12)
            feed(d, NO_HAND, 8)
            feed(d, "O", 12)
            feed(d, NO_HAND, 8)
        assert d.text == "GOGOGO"

    def test_short_gap_between_letters_still_types_once(self):
        d = LetterDebouncer(window=10, min_votes=5, min_confidence=0.5,
                            release_votes=3)
        feed(d, "A", 10)
        feed(d, NO_HAND, 4)   # barely enough to release
        feed(d, "B", 10)
        assert d.text == "AB"

    def test_deliberate_double_letter_still_works(self):
        """The fix must not break repeating a letter on purpose."""
        d = LetterDebouncer(window=10, min_votes=5, min_confidence=0.5,
                            release_votes=4)
        feed(d, "L", 12)
        feed(d, NO_HAND, 12)
        feed(d, "L", 12)
        assert d.text == "LL"


class TestButtonInsert:
    def test_insert_types_immediately(self, deb):
        deb.insert(SPACE)
        deb.insert("X")
        assert deb.text == " X"

    def test_insert_does_not_block_the_next_sign(self, deb):
        feed(deb, "A", 10)
        deb.insert(SPACE)
        feed(deb, "A", 10)
        assert deb.text == "A A"

    def test_configure_keeps_typed_text(self, deb):
        feed(deb, "A", 10)
        deb.configure(window=20, min_votes=15, min_confidence=0.5)
        assert deb.text == "A"
        assert deb.window == 20 and deb.min_votes == 15

    def test_configure_rejects_impossible_settings(self, deb):
        with pytest.raises(ValueError):
            deb.configure(window=5, min_votes=9, min_confidence=0.5)
