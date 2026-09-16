"""Tests for practice scoring."""

import pytest

from practice import PracticeSession, expected_label
from smoothing import SPACE


class FakeClock:
    """A clock the test drives by hand, so timings are exact."""
    def __init__(self): self.t = 100.0
    def __call__(self): return self.t
    def advance(self, seconds): self.t += seconds


@pytest.fixture
def clock():
    return FakeClock()


def spell(session, labels, clock=None, seconds=1.0):
    out = []
    for label in labels:
        if clock:
            clock.advance(seconds)
        out.append(session.observe(label))
    return out


class TestValidation:
    @pytest.mark.parametrize("bad", ["", "   ", "HI!", "one two three four five six"])
    def test_rejects_unusable_targets(self, bad):
        with pytest.raises(ValueError):
            PracticeSession(bad)

    @pytest.mark.parametrize("word", ["JAM", "ZOO", "jazz"])
    def test_rejects_motion_letters(self, word):
        with pytest.raises(ValueError, match="motion"):
            PracticeSession(word)

    def test_normalises_case_and_spacing(self):
        assert PracticeSession("  hi   there ").target == "HI THERE"


class TestWalkthrough:
    def test_correct_letters_advance(self, clock):
        s = PracticeSession("CAT", clock)
        assert s.current == "C"
        assert spell(s, ["C", "A", "T"], clock) == ["correct"] * 3
        assert s.done and s.current is None

    def test_wrong_letter_does_not_advance(self, clock):
        """Advancing on an error would make every later letter wrong too."""
        s = PracticeSession("CAT", clock)
        assert s.observe("A") == "wrong"
        assert s.current == "C"          # still waiting for C
        assert s.observe("C") == "correct"
        assert s.current == "A"

    def test_space_is_expected_as_the_space_label(self, clock):
        s = PracticeSession("HI YOU", clock)
        spell(s, ["H", "I"], clock)
        assert s.expected == SPACE
        assert s.observe(SPACE) == "correct"

    def test_input_after_finishing_is_ignored(self, clock):
        s = PracticeSession("A", clock)
        spell(s, ["A"], clock)
        assert s.observe("B") == "ignored"

    def test_typed_tracks_progress(self, clock):
        s = PracticeSession("CAT", clock)
        spell(s, ["C", "A"], clock)
        assert s.typed() == "CA"

    def test_skip_moves_on_and_is_recorded(self, clock):
        s = PracticeSession("CAT", clock)
        s.skip()
        assert s.current == "A"
        assert s.summary()["skipped"] == 1


class TestScoring:
    def test_clean_run_scores_full_marks(self, clock):
        s = PracticeSession("CAT", clock)
        spell(s, ["C", "A", "T"], clock, seconds=2.0)
        r = s.summary()
        assert r["first_try"] == 3
        assert r["first_try_rate"] == 1.0
        assert r["accuracy"] == 1.0
        assert r["seconds_per_letter"] == 2.0

    def test_mistakes_lower_accuracy_but_not_completion(self, clock):
        s = PracticeSession("CAT", clock)
        spell(s, ["A", "S", "C", "A", "T"], clock)   # two misses on C
        r = s.summary()
        assert r["done"] is True
        assert r["attempts"] == 5
        assert r["accuracy"] == pytest.approx(3 / 5)
        assert r["first_try"] == 2                    # A and T were clean

    def test_records_what_each_letter_was_mistaken_for(self, clock):
        s = PracticeSession("CAT", clock)
        spell(s, ["A", "A", "C", "A", "T"], clock)
        assert s.summary()["confusions"][0] == ["C", "A", 2]

    def test_per_letter_timing_is_measured_separately(self, clock):
        s = PracticeSession("AB", clock)
        clock.advance(1.0); s.observe("A")
        clock.advance(4.0); s.observe("B")
        detail = s.summary()["detail"]
        assert detail[0]["seconds"] == 1.0
        assert detail[1]["seconds"] == 4.0

    def test_elapsed_stops_when_the_word_is_finished(self, clock):
        s = PracticeSession("A", clock)
        clock.advance(3.0); s.observe("A")
        clock.advance(60.0)                # user wanders off
        assert s.summary()["seconds"] == 3.0

    def test_partial_run_reports_only_what_was_attempted(self, clock):
        s = PracticeSession("CAT", clock)
        spell(s, ["C"], clock)
        r = s.summary()
        assert r["done"] is False
        assert r["completed"] == 1 and r["letters"] == 3


def test_expected_label_maps_space():
    assert expected_label(" ") == SPACE
    assert expected_label("a") == "A"