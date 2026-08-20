"""Offline stand-ins for BoundarySelector, so tests never call a real model.

Same pattern as any other external-dependency seam in this project: the real
implementation sits behind a Protocol (era.edgar.boundaries.BoundarySelector)
so tests can swap in something deterministic and free.
"""

from era.edgar.boundaries import Boundary, HeadingCandidate


class FirstMatchSelector:
    """Picks the first candidate per item and runs to the next candidate.

    Deliberately naive -- it reproduces the pre-Task-19 regex behaviour
    (first occurrence wins, no judgment about what kind of line it is), so
    tests can assert what the real selector must do better, and can assert
    that era.edgar.sections' verification step catches this selector's
    mistakes rather than trusting them.
    """

    def select(self, candidates: list[HeadingCandidate]) -> dict[str, Boundary]:
        chosen: dict[str, Boundary] = {}
        for candidate in candidates:
            if candidate.item in chosen:
                continue
            following = next((c.index for c in candidates if c.index > candidate.index), None)
            chosen[candidate.item] = Boundary(candidate.index, following)
        return chosen


class ScriptedSelector:
    """Returns boundaries a test dictates, so slicing and guards can be tested alone."""

    def __init__(self, boundaries: dict[str, Boundary]) -> None:
        self._boundaries = boundaries

    def select(self, candidates: list[HeadingCandidate]) -> dict[str, Boundary]:
        return self._boundaries
