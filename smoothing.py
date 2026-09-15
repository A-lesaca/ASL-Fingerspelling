from __future__ import annotations

from collections import Counter, deque
from typing import Deque, Optional, Tuple

NO_HAND = "__none__"

SPACE = "space"
DELETE = "del"


class LetterDebouncer:
    """Collapse a stream of frame predictions into discrete committed letters. """

    def __init__(
        self,
        window: int = 12,
        min_votes: int = 9,
        min_confidence: float = 0.80,
        release_votes: int = 5,
    ) -> None:
        if not 0 < min_votes <= window:
            raise ValueError("min_votes must be in (0, window]")
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be in [0, 1]")
        if release_votes <= 0:
            raise ValueError("release_votes must be positive")

        self.window = window
        self.min_votes = min_votes
        self.min_confidence = min_confidence
        self.release_votes = release_votes

        self._frames: Deque[str] = deque(maxlen=window)
        self._latched: Optional[str] = None  
        self._away = 0  
        self._chars: list[str] = []


    @property
    def text(self) -> str:
        """The text typed so far."""
        return "".join(self._chars)

    @property
    def latched(self) -> Optional[str]:
        return self._latched

    def progress(self) -> Tuple[Optional[str], float]:
        """Leading candidate and its vote fraction, for an on-screen progress bar."""
        if not self._frames:
            return None, 0.0
        label, votes = Counter(self._frames).most_common(1)[0]
        if label == NO_HAND:
            return None, 0.0
        return label, votes / self.min_votes

    def update(self, label: str, confidence: float = 1.0) -> Optional[str]:
        """Feed one frame's prediction."""
        if confidence < self.min_confidence:
            label = NO_HAND
        self._frames.append(label)

        # Track time spent away from the latched letter so it can be released.
        if self._latched is not None:
            if label != self._latched:
                self._away += 1
                if self._away >= self.release_votes:
                    released = self._latched
                    self._latched = None
                    self._away = 0
                    # Discard the released letter's frames. Releasing only
                    # clears the guard against re-committing; the window can
                    # still be full of the sign the hand just left, and on a
                    # short release those stale frames are enough to win the
                    # vote again the instant the guard drops -- typing the
                    # letter twice. Requiring fresh frames fixes that without
                    # penalising a genuine move to a different letter, whose
                    # frames are kept.
                    self._frames = deque(
                        (f for f in self._frames if f != released),
                        maxlen=self.window,
                    )
            else:
                self._away = 0

        winner, votes = Counter(self._frames).most_common(1)[0]
        if winner == NO_HAND or votes < self.min_votes:
            return None
        if winner == self._latched:
            return None 

        self._latched = winner
        self._away = 0
        self._apply(winner)
        return winner

    def reset(self) -> None:
        """Clear typed text and all internal state."""
        self._frames.clear()
        self._latched = None
        self._away = 0
        self._chars.clear()

    def _apply(self, label: str) -> None:
        if label == SPACE:
            self._chars.append(" ")
        elif label == DELETE:
            if self._chars:
                self._chars.pop()
        else:
            self._chars.append(label)
    def insert(self, label: str) -> None:
        """Type a character directly, bypassing the voting.

        The on-screen buttons need this: a click is a deliberate instruction,
        not a noisy frame prediction, so it should take effect immediately
        rather than waiting for a majority. Latching is cleared so the next
        held sign still commits normally.
        """
        self._apply(label)
        self._latched = None
        self._away = 0
        self._frames.clear()

    def configure(self, window: int, min_votes: int, min_confidence: float) -> None:
        """Re-apply the voting settings, keeping any text already typed."""
        if not 0 < min_votes <= window:
            raise ValueError("min_votes must be in (0, window]")
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be in [0, 1]")
        self.window = window
        self.min_votes = min_votes
        self.min_confidence = min_confidence
        self._frames = deque(self._frames, maxlen=window)
        self._latched = None
        self._away = 0
