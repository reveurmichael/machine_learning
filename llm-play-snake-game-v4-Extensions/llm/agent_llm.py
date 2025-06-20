"""LLM-powered policy that conforms to the core.game_agents.SnakeAgent protocol.

This tiny wrapper lets the existing prompt / parsing pipeline live *entirely*
inside the `llm` package so the rest of the codebase can treat it like any
other agent.  There are **zero** imports from `llm` back into `core` – the
dependency direction now flows one way.

As LLM is Task0 specific, this whole module is Task0 specific.
"""

from __future__ import annotations

from typing import Any, Optional

from core.game_agents import SnakeAgent
from llm.client import LLMClient

# --------------------------
# Public class
# --------------------------

class LLMSnakeAgent(SnakeAgent):
    """A pluggable agent that queries a large-language model for each move."""

    def __init__(
        self,
        provider: str = "hunyuan",
        model: Optional[str] = None,
        **generation_kwargs: Any,
    ) -> None:
        """Create a new LLM-backed agent.

        Parameters
        ----------
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

        self._client = LLMClient(provider=provider, model=model)
        self._gen_kwargs = generation_kwargs

    # ------------------
    # SnakeAgent implementation
    # ------------------

    def get_move(self, game: Any) -> str | None:  # type: ignore[override]
        """Return the next direction proposed by the LLM.

        The `game` object is expected to expose the same public surface as
        ``core.game_logic.GameLogic`` – namely:
            • ``get_state_representation()``
            • ``parse_llm_response(str) -> str | None``
        The second call mutates the game to set ``planned_moves`` and related
        diagnostics exactly like the old code path, so UI elements continue to
        work with *zero* changes.
        """

        # Derive prompt from the current board state
        try:
            prompt: str = game.get_state_representation()  # type: ignore[attr-defined]
        except AttributeError as exc:  # pragma: no cover – dev error
            raise TypeError(
                "LLMSnakeAgent expects a game object with 'get_state_representation()'"
            ) from exc

        # Call the LLM synchronously (network latency dominates anyway)
        response: str = self._client.generate_response(prompt, **self._gen_kwargs)

        # Use the existing helper to translate raw text into a move &
        # side-effect game state (planned moves, pretty reasoning, …)
        try:
            move = game.parse_llm_response(response)  # type: ignore[attr-defined]
        except AttributeError:
            # Fallback if the game object does *not* embed the legacy parser.
            # In that case we attempt a minimal parse inline.
            move = _simple_parse(response)

        return move


# --------------------------
# Fallback helpers – only used when the game object lacks parse_llm_response
# --------------------------

def _simple_parse(llm_response: str) -> str | None:
    """Best-effort JSON ‹moves› extraction without full GameLogic context."""

    from json import loads

    try:
        data = loads(llm_response)
        moves = data.get("moves") if isinstance(data, dict) else None
        if moves and isinstance(moves, list):
            return moves[0]
    except Exception:  # noqa: BLE001 – we really want to swallow *anything*
        pass
    return None 