from typing import Iterable, List, Union

DIRECTION_STR = Union[str, int, float]  # loose – we only care for strings


def normalize_direction(move: DIRECTION_STR):
    """Return a canonical representation of a single direction token.

    * Ignore *non-string* values (they pass through unchanged – defensive).
    * Strip whitespace and convert to upper-case so that "right", "Right\n" → "RIGHT".
    """
    if isinstance(move, str):
        return move.strip().upper()
    return move


def normalize_directions(moves: Iterable[DIRECTION_STR]) -> List[DIRECTION_STR]:
    """Apply *normalize_direction* element-wise to any iterable of moves."""
    return [normalize_direction(m) for m in moves] 