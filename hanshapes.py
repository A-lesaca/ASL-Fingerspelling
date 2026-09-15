"""Plain-language descriptions of the static ASL fingerspelling handshapes.

Written out rather than shipped as pictures on purpose: alphabet charts found
online are copyrighted illustrations, and an inaccurate one would teach the
wrong pose and end up baked into the training data. These describe the
right-hand shape as seen by someone signing, palm generally toward the viewer.

J and Z are absent because both are drawn with movement.
"""

from __future__ import annotations

HANDSHAPES: dict[str, str] = {
    "A": "Closed fist, thumb straight up alongside your index finger.",
    "B": "Flat hand, fingers straight up and together, thumb folded across the palm.",
    "C": "Fingers and thumb curved into the shape of a C.",
    "D": "Index finger straight up, other fingertips meeting the thumb in a circle.",
    "E": "Fingers curled down, fingertips resting on the thumb.",
    "F": "Thumb and index finger touching in a circle, other three fingers straight up.",
    "G": "Index finger and thumb extended parallel, pointing sideways.",
    "H": "Index and middle fingers extended together, pointing sideways.",
    "I": "Fist with only the little finger straight up.",
    "K": "Index and middle fingers up in a V, thumb pressed between them.",
    "L": "Index finger up and thumb out, making an L.",
    "M": "Thumb tucked under the index, middle and ring fingers folded over it.",
    "N": "Thumb tucked under the index and middle fingers folded over it.",
    "O": "Fingers curved round to meet the thumb, making an O.",
    "P": "The K shape rotated so the fingers point downward.",
    "Q": "The G shape rotated so the finger and thumb point downward.",
    "R": "Index and middle fingers crossed, pointing up.",
    "S": "Closed fist with the thumb across the front of the fingers.",
    "T": "Fist with the thumb poking up between the index and middle fingers.",
    "U": "Index and middle fingers straight up, held together.",
    "V": "Index and middle fingers straight up, spread apart.",
    "W": "Index, middle and ring fingers straight up and spread.",
    "X": "Fist with the index finger crooked into a hook.",
    "Y": "Thumb and little finger extended, the rest folded down.",
    "space": "Pick any spare pose you can hold — an open palm works well.",
    "del": "Pick another spare pose, clearly different from your space sign.",
}


def describe(label: str) -> str:
    """Return guidance for a label, or a prompt for custom gestures."""
    if label in HANDSHAPES:
        return HANDSHAPES[label]
    return f"Hold the still pose you want to mean \u201c{label}\u201d."