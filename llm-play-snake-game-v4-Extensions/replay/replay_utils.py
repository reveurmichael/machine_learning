from __future__ import annotations

"""Utility helpers for the *replay* package.

Pure functions that do not depend on `ReplayEngine` internal state.  They keep
I/O-heavy code out of the engine class while allowing it to focus on gameplay.
"""

from pathlib import Path
import json
from typing import Any, Dict, Optional, Tuple

from utils.file_utils import get_game_json_filename, join_log_path

# Internal models (kept in a separate file for cleaner imports)
from replay.replay_data import ReplayDataLLM


def load_game_json(log_dir: str, game_number: int) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Return the path and decoded JSON dict for *game_number*.

    Parameters
    ----------
    log_dir
        Directory containing the `game_*.json` artefacts.
    game_number
        1-based index of the desired game file.

    Returns
    -------
    (file_path, data) where *data* is ``None`` when the file is missing or
    cannot be parsed.
    """
    game_filename = get_game_json_filename(game_number)
    game_file = join_log_path(log_dir, game_filename)

    file_path = Path(game_file)
    if not file_path.exists():
        print(f"[replay] Game {game_number} data not found in {log_dir}")
        return str(file_path), None

    try:
        with file_path.open("r", encoding="utf-8") as f:
            data: Dict[str, Any] = json.load(f)
        return str(file_path), data
    except Exception as exc:  # pragma: no cover – defensive catch
        print(f"[replay] Error reading {file_path}: {exc}")
        return str(file_path), None


def parse_game_data(game_data: Dict[str, Any]) -> Optional[ReplayDataLLM]:
    """Return a lightweight, replay-friendly view of a *game_N.json* blob.

    Parameters
    ----------
    game_data
        Raw dictionary obtained by ``json.load``-ing a *game_N.json* file. The
        function does **not** touch the file-system; the caller is expected to
        have already read the JSON from disk (see :func:`load_game_json`).

    What the function does
    ----------------------
    1. Validates that the mandatory ``detailed_history`` block is present – if
       it is missing we consider the file corrupted and return *None* so the
       caller can skip the game gracefully.
    2. Extracts the two time-series that the replay engine absolutely needs:
       ``apple_positions`` and ``moves``.  These arrays are validated for
       minimal sanity (non-empty; apples have two coordinates).  The raw lists
       are **not** mutated, so the caller keeps full fidelity.
    3. Deduces *planned moves* from the first round in ``rounds_data`` when
       available.  These planned moves are purely cosmetic (shown in the web
       sidebar) and the function tolerates any formatting quirks by falling
       back to an empty list.
    4. Builds human-readable strings for *primary* and *parser* LLMs so that
       the GUI can display something like ``"ollama/deepseek-14b"`` without
       duplicating string-building logic.

    Returns
    -------
    ReplayDataLLM | None
        A dataclass instance containing the fields required by
        :class:`replay.replay_engine.ReplayEngine`.  If any critical component
        is missing or malformed the function prints a short diagnostic message
        and returns *None*, leaving error-handling to the caller.
    """
    detailed = game_data.get("detailed_history")
    if not isinstance(detailed, dict):
        print("[replay] detailed_history missing")
        return None

    apples: list[list[int]] = []
    for pos in detailed.get("apple_positions", []):
        if isinstance(pos, dict) and {"x", "y"}.issubset(pos):
            apples.append([pos["x"], pos["y"]])
        elif isinstance(pos, (list, tuple)) and len(pos) == 2:
            apples.append(list(pos))
    if not apples:
        print("[replay] empty apple_positions")
        return None

    moves: list[str] = detailed.get("moves", [])
    if not moves:
        print("[replay] empty moves array")
        return None

    planned: list[str] = []
    rounds = detailed.get("rounds_data", {})
    if rounds:
        try:
            first_key = sorted(rounds, key=lambda k: int(k.split("_")[1]))[0]
            first_round = rounds[first_key]
            if isinstance(first_round.get("moves"), list) and len(first_round["moves"]) > 1:
                planned = first_round["moves"][1:]
        except Exception:
            pass

    llm_info = game_data.get("llm_info", {})
    primary_llm = f"{llm_info.get('primary_provider', 'Unknown')}/{llm_info.get('primary_model', 'Unknown')}"
    secondary_llm = "None/None"
    if llm_info.get("parser_provider") and str(llm_info.get("parser_provider")).lower() != "none":
        secondary_llm = f"{llm_info.get('parser_provider')}/{llm_info.get('parser_model')}"

    return ReplayDataLLM(
        apple_positions=apples,
        moves=moves,
        planned_moves=planned,
        game_end_reason=game_data.get("game_end_reason"),
        primary_llm=primary_llm,
        secondary_llm=secondary_llm,
        timestamp=game_data.get("metadata", {}).get("timestamp"),
        llm_response=game_data.get("llm_response"),
        full_json=game_data,
    )
