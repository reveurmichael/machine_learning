"""Task-0 LLM agent (`LLMSnakeAgent`).

This module houses **all** heavy-weight communication with language models –
prompt construction, dual-LLM parsing, disk logging, token/time statistics,
continuation bookkeeping – and exposes it as a normal
:pyclass:`core.game_agents.SnakeAgent` implementation.  Moving that logic out
of *core* achieves two design goals:

1. **Pluggability** – The game loop now talks to an *agent* instead of a
   hard-coded helper.  Future heuristic / RL / curriculum tasks can supply
   their own agents without touching the loop.

2. **Unidirectional dependency graph** – Only the :pymod:`llm` package knows
   about language-model providers.  Core gameplay code (`core.*`) remains
   completely provider-agnostic.

While the class is Task-0 specific for now, it sets the precedent for richer
agents in Tasks 1–5 (e.g. `HeuristicSnakeAgent`, `RlSnakeAgent`).
"""

from __future__ import annotations

from typing import Any, Optional, TYPE_CHECKING

from core.game_agents import SnakeAgent
from llm.client import LLMClient

if TYPE_CHECKING:  # pragma: no cover – avoid runtime import cycle
    from core.game_manager import GameManager

# ---------------------
# Public class
# ---------------------

class LLMSnakeAgent(SnakeAgent):
    """A pluggable agent that queries a large-language model for each move."""

    def __init__(
        self,
        manager: "GameManager | None" = None,
        provider: str = "ollama",
        model: Optional[str] = None,
        **generation_kwargs: Any,
    ) -> None:
        """Create a new LLM-backed agent.

        Parameters
        ----------
        manager:
            Reference to the full GameManager.
        provider:
            Identifier understood by :class:`llm.client.LLMClient` (e.g.
            ``"openai"``, ``"deepseek"``).
        model:
            Concrete model name to forward to the provider (can be ``None`` to
            accept the provider default).
        **generation_kwargs:
            Extra key-word arguments passed verbatim to
            :py:meth:`LLMClient.generate_response` (e.g. ``temperature``).
        """

        self._manager = manager

        # Decide which LLMClient to use
        if manager is not None and getattr(manager, "llm_client", None) is not None:
            self._client = manager.llm_client  # type: ignore[attr-defined]
        else:
            self._client = LLMClient(provider=provider, model=model)
            if manager is not None:
                manager.llm_client = self._client  # share for consistency

        self._gen_kwargs = generation_kwargs

    # ------------------
    # SnakeAgent implementation
    # ------------------

    def get_move(self, game: Any) -> str | None:  # type: ignore[override]
        """Return the next direction proposed by the LLM.

        The agent will first consume any existing planned moves. Only when
        the plan is exhausted will it request a new plan from the LLM and
        increment the round counter.
        """

        # First, try to consume any existing planned moves
        if hasattr(game, "planned_moves") and game.planned_moves:
            if hasattr(game, "get_next_planned_move"):
                return game.get_next_planned_move()
            # Fallback for game objects without get_next_planned_move method
            return game.planned_moves.pop(0)

        # No planned moves available - need to request a new plan from LLM
        if self._manager is not None:
            from llm.communication_utils import get_llm_response  # local import

            game_manager = self._manager

            # Increment the round counter *before* querying the LLM
            # A round begins when we ask for a new plan
            if getattr(game_manager, "_first_plan", False):
                game_manager._first_plan = False  # first round is already #1
            else:
                game_manager.increment_round("new plan requested by agent")

            # Mark awaiting_plan to keep UI behaviour identical
            game_manager.awaiting_plan = True
            move, game_active = get_llm_response(
                game_manager, round_id=game_manager.round_count
            )  # type: ignore[arg-type]
            game_manager.awaiting_plan = False

            # Update game manager state
            game_manager.need_new_plan = False
            game_manager.game_active = game_active

            return move

        # ----------------
        # Fallback standalone mode (no GameManager) – original lightweight path
        # ----------------

        # Derive prompt from the current board state
        try:
            prompt: str = game.get_state_representation()  # type: ignore[attr-defined]
        except AttributeError as exc:  # pragma: no cover – dev error
            raise TypeError(
                "LLMSnakeAgent expects a game object with 'get_state_representation()'"
            ) from exc

        response: str = self._client.generate_response(prompt, **self._gen_kwargs)

        try:
            move = game.parse_llm_response(response)  # type: ignore[attr-defined]
        except AttributeError:
            move = _simple_parse(response)

        return move


# ---------------------
# Fallback helpers – only used when the game object lacks parse_llm_response
# ---------------------

def _simple_parse(llm_response: str) -> str | None:
    """Very lightweight JSON extractor used *only* as a last-resort fallback.

    Motivation
    ----------
    ``LLMSnakeAgent`` is usually instantiated **with** a
    :class:`core.game_manager.GameManager`, in which case the heavy parsing
    pipeline located in :pymod:`llm.communication_utils` runs and this helper
    is never touched.

    During *unit tests*, quick demos or in external notebooks developers often
    want to call the agent in complete isolation:

    >>> agent = LLMSnakeAgent()                # no GameManager passed
    >>> move  = agent.get_move(minimal_game)  # minimal stub object

    Such a stub usually provides *only* ``get_state_representation()`` and
    not the full ``parse_llm_response()`` helper that Task-0ʼs
    :class:`core.game_logic.GameLogic` implements.  When that method is
    missing ``LLMSnakeAgent`` drops down to ``_simple_parse`` so that we still
    obtain a **single** move instead of ``None`` – avoiding an EMPTY sentinel
    and keeping the test concise.

    Overlap with utils.json_utils
    ---------------------
    The robust JSON repair / validation logic lives in
    :pymod:`utils.json_utils`.  Re-using it here would pull heavy regular‐
    expression machinery into the tight loop of every *test* invocation where
    simplicity is preferred over full correctness.  Hence we deliberately
    re-implement a *tiny* subset:

    * Look for a JSON object or array at the top level.
    * Return the very first string inside ``"moves"`` when present.

    If anything fails we silently return ``None`` so the caller can treat it
    as an EMPTY move.

    Performance / maintenance cost
    ---------------------
    The function is ~10 LOC, uses only the standard library, and is executed
    **only** in the aforementioned edge cases, so it has negligible impact on
    runtime and maintenance.
    """

    from json import loads

    try:
        data = loads(llm_response)
        moves = data.get("moves") if isinstance(data, dict) else None
        if moves and isinstance(moves, list):
            return moves[0]
    except Exception:  # noqa: BLE001 – we really want to swallow *anything*
        pass
    return None 