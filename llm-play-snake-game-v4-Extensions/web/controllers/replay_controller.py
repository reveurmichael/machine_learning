"""
Replay Controller - MVC Architecture
--------------------

Controller for game replay functionality.
Handles replay navigation, playback control, and analytics display.

Design Patterns Used:
    - Template Method: Inherits viewing request handling from GameViewingController
    - Strategy Pattern: Different replay data sources (file, database, memory)
    - Command Pattern: Replay navigation commands
    - Observer Pattern: Monitors replay state changes

Educational Goals:
    - Show how inheritance enables specialized functionality
    - Demonstrate separation of replay logic from general viewing logic
    - Illustrate how controllers can orchestrate complex replay workflows

Naming convention reminder:
    • Task-0 concrete viewer; therefore the short name *ReplayController*.
    • Generic/shared behaviour lives in `BaseGameViewingController`.
    • Extensions provide their own viewer subclasses (e.g.
      `HeuristicReplayController`) inside their `extensions/` namespace and
      inherit from the *base*.
"""

import logging
from typing import Dict, Any, Optional, List

from .game_controllers import BaseGameViewingController
from .base_controller import RequestContext
from ..models import GameStateModel
from ..views import WebViewRenderer

logger = logging.getLogger(__name__)


class ReplayController(BaseGameViewingController):
    """
    Controller for game replay functionality.
    
    Extends BaseGameViewingController to provide replay-specific features.
    Handles replay loading, navigation, and analytics.
    """
    
    def __init__(self, model_manager: GameStateModel, view_renderer: WebViewRenderer, **kwargs):
        """Initialize replay controller."""
        super().__init__(model_manager, view_renderer, viewing_mode='replay', **kwargs)
        
        # Replay-specific configuration
        self.current_replay_id: Optional[str] = None
        self.replay_frames: List = []
        
        logger.info("Initialized ReplayController")
    
    def _handle_viewing_action(self, action: str, context: RequestContext) -> Dict[str, Any]:
        """Handle replay-specific actions."""
        return {
            'success': True,
            'message': f'Replay action: {action}'
        }
    
    def _get_analytics(self) -> Dict[str, Any]:
        """Get analytics data for the replay session."""
        return {
            'message': 'Replay analytics placeholder'
        }

    # ---------------------
    # MVC Index rendering hooks – ensure the correct Jinja template is used
    # instead of the generic fallback defined in *BaseWebController*.
    # ---------------------

    def get_index_template_name(self) -> str:  # noqa: D401 – simple override
        """Return the Jinja template that renders the replay UI."""

        return "replay.html"

    def get_index_template_context(self) -> Dict[str, Any]:  # noqa: D401
        """Populate the template context with basic metadata for the header.

        The heavy-lifting (actual state updates) is done via the `/api/state`
        endpoint, fetched asynchronously by the front-end JavaScript.  Here we
        merely seed some placeholders so the initial page loads without the
        fallback warning banner.
        """

        return {
            "title": "Snake Game – Replay",
            "controller_name": self.__class__.__name__,
            "game_mode": "replay",
        }

    # ---------------------
    # API: enrich the state payload so the front-end can show progress, speed
    # and LLM meta data (JS expects these fields at *top level*).
    # ---------------------

    def handle_state_request(self, context: RequestContext) -> Dict[str, Any]:  # type: ignore[override]
        """Return the current replay state with Task-0 extras."""

        # Get the generic representation first.
        state = super().handle_state_request(context)

        try:
            eng = self.model_manager.state_provider.replay_engine  # type: ignore[attr-defined]

            # Inject additional attributes consumed by `replay.js`.
            state.update(
                {
                    "game_number": getattr(eng, "game_number", 1),
                    "total_games": getattr(eng, "total_games", 1),
                    "move_index": getattr(eng, "move_index", 0),
                    "total_moves": len(getattr(eng, "moves", [])),
                    "game_end_reason": getattr(eng, "game_end_reason", None),
                    "paused": getattr(eng, "paused", False),
                    "primary_llm": getattr(eng, "primary_llm", None),
                    "secondary_llm": getattr(eng, "secondary_llm", None),
                    "pause_between_moves": getattr(eng, "pause_between_moves", 1.0),
                    "speed": (
                        1.0 / eng.pause_between_moves if eng.pause_between_moves else 1.0
                    ),
                }
            )

        except Exception as exc:  # pragma: no cover – defensive guard
            logger.debug(f"ReplayController: failed to enrich state – {exc}")

        return state

    # ---------------------
    # Control endpoint – the legacy front-end sends {command: "play" | …}
    # while the new MVC base class expects {action: ...}.  We translate here
    # so we don't have to touch the JavaScript.
    # ---------------------

    def handle_control_request(self, context: RequestContext) -> Dict[str, Any]:  # type: ignore[override]
        data = context.data or {}
        command = data.get("command")

        eng = self.model_manager.state_provider.replay_engine  # type: ignore[attr-defined]

        if command == "play":
            eng.paused = False
            return {"status": "playing"}
        if command == "pause":
            eng.paused = True
            return {"status": "paused"}
        if command == "next_game":
            eng.game_number += 1
            if not eng.load_game_data(eng.game_number):
                eng.game_number -= 1
                return {"status": "error", "message": "No next game"}
            return {"status": "ok"}
        if command == "prev_game":
            if eng.game_number > 1:
                eng.game_number -= 1
                eng.load_game_data(eng.game_number)
                return {"status": "ok"}
            return {"status": "error", "message": "Already at first game"}
        if command == "restart_game":
            eng.load_game_data(eng.game_number)
            return {"status": "ok"}
        if command == "speed_up":  # speeds playback (decrease pause)
            # Mirror the PyGame hotkey logic: multiply by 0.75 (up-key in desktop replay)
            eng.pause_between_moves = max(0.1, eng.pause_between_moves * 0.75)
            return {
                "status": "ok",
                "pause_between_moves": eng.pause_between_moves,
                "speed": 1.0 / eng.pause_between_moves,
            }
        if command == "speed_down":  # slows playback (increase pause)
            # Mirror PyGame logic: multiply by 1.25 (down-key in desktop replay)
            eng.pause_between_moves = min(3.0, eng.pause_between_moves * 1.25)
            return {
                "status": "ok",
                "pause_between_moves": eng.pause_between_moves,
                "speed": 1.0 / eng.pause_between_moves,
            }

        # Fallback to base implementation so future MVC actions still work.
        context.data["action"] = command
        return super().handle_control_request(context)
