"""Scoring for practice runs: spell a target word, measure how it went.

Kept separate from the camera and the web server so the scoring rules can be
tested directly. The session is fed committed letters -- the output of the
debouncer, not raw per-frame predictions -- so one entry here corresponds to
one letter the user actually produced.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from smoothing import SPACE

# What a target character expects as a committed label.
def expected_label(char: str) -> str:
    return SPACE if char == " " else char.upper()


@dataclass
class LetterResult:
    target: str
    attempts: int = 0
    seconds: float = 0.0
    wrong: list[str] = field(default_factory=list)
    skipped: bool = False

    @property
    def first_try(self) -> bool:
        return self.attempts == 1 and not self.wrong and not self.skipped


class PracticeSession:
    """Walks through a target word, scoring each letter as it is signed.

    A wrong letter does not advance the session: the user stays on the same
    target and tries again. That is deliberate -- advancing on an error would
    desynchronise the rest of the word and make every later letter look wrong
    too, which tells you nothing about which handshape is actually the problem.
    """

    def __init__(self, target: str, clock: Callable[[], float] = time.monotonic):
        cleaned = " ".join(target.upper().split())
        if not cleaned:
            raise ValueError("Type a word to practise.")
        if len(cleaned) > 24:
            raise ValueError("Keep practice words to 24 characters or fewer.")
        bad = {c for c in cleaned if c != " " and not c.isalpha()}
        if bad:
            raise ValueError("Use letters and spaces only.")
        motion = {c for c in cleaned if c in {"J", "Z"}}
        if motion:
            raise ValueError(
                f"{', '.join(sorted(motion))} need motion and are not modelled."
            )

        self.target = cleaned
        self._clock = clock
        self._index = 0
        self._started = clock()
        self._letter_started = self._started
        self.finished_at: Optional[float] = None
        self.results: list[LetterResult] = [LetterResult(c) for c in cleaned]

    # -- state -------------------------------------------------------------

    @property
    def done(self) -> bool:
        return self._index >= len(self.target)

    @property
    def current(self) -> Optional[str]:
        """The character still to be signed, or None once finished."""
        return None if self.done else self.target[self._index]

    @property
    def expected(self) -> Optional[str]:
        c = self.current
        return None if c is None else expected_label(c)

    @property
    def elapsed(self) -> float:
        end = self.finished_at if self.finished_at is not None else self._clock()
        return end - self._started

    def typed(self) -> str:
        """What the user has completed so far."""
        return self.target[: self._index]

    # -- input -------------------------------------------------------------

    def observe(self, label: str) -> str:
        """Feed one committed letter. Returns 'correct', 'wrong' or 'ignored'."""
        if self.done:
            return "ignored"
        result = self.results[self._index]
        result.attempts += 1

        if label == self.expected:
            result.seconds = self._clock() - self._letter_started
            self._advance()
            return "correct"

        result.wrong.append(label)
        return "wrong"

    def skip(self) -> None:
        """Give up on the current letter and move on."""
        if self.done:
            return
        result = self.results[self._index]
        result.skipped = True
        result.seconds = self._clock() - self._letter_started
        self._advance()

    def _advance(self) -> None:
        self._index += 1
        self._letter_started = self._clock()
        if self.done:
            self.finished_at = self._clock()

    # -- output ------------------------------------------------------------

    def summary(self) -> dict:
        completed = [r for r in self.results if r.attempts or r.skipped]
        scored = [r for r in completed if not r.skipped]
        first_try = sum(1 for r in completed if r.first_try)

        # Which handshape was mistaken for which. This is the measurement worth
        # reporting: it is taken live, from the user's own signing, rather than
        # from a held-out split of the training data.
        confusions: dict[tuple[str, str], int] = {}
        for r in self.results:
            for got in r.wrong:
                key = (r.target, got)
                confusions[key] = confusions.get(key, 0) + 1

        total_attempts = sum(r.attempts for r in completed)
        return {
            "target": self.target,
            "done": self.done,
            "letters": len(self.target),
            "completed": len(completed),
            "first_try": first_try,
            "first_try_rate": first_try / len(completed) if completed else 0.0,
            "attempts": total_attempts,
            "accuracy": len(scored) / total_attempts if total_attempts else 0.0,
            "skipped": sum(1 for r in self.results if r.skipped),
            "seconds": round(self.elapsed, 2),
            "seconds_per_letter": round(
                sum(r.seconds for r in scored) / len(scored), 2
            ) if scored else 0.0,
            "detail": [
                {
                    "target": r.target,
                    "attempts": r.attempts,
                    "seconds": round(r.seconds, 2),
                    "wrong": r.wrong,
                    "skipped": r.skipped,
                }
                for r in self.results
            ],
            "confusions": sorted(
                ([t, g, n] for (t, g), n in confusions.items()),
                key=lambda row: -row[2],
            )[:8],
        }