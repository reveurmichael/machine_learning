## BaseReplayEngine

In the python file "./replay/replay_engine.py", we have the class ReplayEngine. It should be used by task0, 1, 2, 3, 4, 5.

```python
class BaseReplayEngine(GameController):
    """Headless-capable board replay engine.

    The base class contains only the state and helpers that are independent of
    Task-0 specifics (no log-file I/O, no LLM metadata).  Future tasks can
    inherit from it to implement custom playback logic while reusing the
    common event loop and pause handling.
    """

    # ------------------
    # Construction & basic state
    # ------------------

    def __init__(
        self,
        log_dir: str,
        move_pause: float = 1.0,
        auto_advance: bool = False,
        use_gui: bool = True,
    ) -> None:  # noqa: D401 – simple init
        super().__init__(use_gui=use_gui)

        # Attached GUI (pygame or web proxy).  May be ``None`` for headless.
        self.gui = None

        # Main loop flags
        self.running: bool = True
        self.paused: bool = False

        self.log_dir: str = log_dir
        self.pause_between_moves: float = move_pause
        self.auto_advance: bool = auto_advance

        # Game state specific to replay
        self.game_number: int = 1
        self.apple_positions: List[List[int]] = []
        self.apple_index: int = 0
        self.moves: List[str] = []
        self.move_index: int = 0
        self.moves_made: List[str] = []
        self.game_stats: Dict[str, Any] = {}

        # Game statistics from the log file
        self.game_end_reason: Optional[str] = None

    # ------------------
    # Extensibility hooks – subclasses are expected to override
    # ------------------

    def load_game_data(self, *args, **kwargs):  # pragma: no cover
        """Load a game into memory (to be implemented by subclasses)."""

        raise NotImplementedError

    def update(self):  # pragma: no cover
        """Advance one frame (to be implemented by subclasses)."""

        raise NotImplementedError

    def handle_events(self):  # pragma: no cover
        """Process user/OS events (to be implemented by subclasses)."""

        raise NotImplementedError

    def run(self):  # pragma: no cover
        """Main replay loop (to be implemented by subclasses)."""

        raise NotImplementedError

    # ------------------
    # Convenience helpers
    # ------------------

    def set_gui(self, gui_instance):  # type: ignore[override]
        """Attach a GUI instance and keep its paused flag in sync."""

        super().set_gui(gui_instance)

        if gui_instance and hasattr(gui_instance, "set_paused"):
            gui_instance.set_paused(self.paused)

    # ------------------
    # Common state builder – shared by GUI implementations (pygame, web, etc.)
    # ------------------

    def _build_state_base(self) -> dict:
        """Return a dict with the *core* replay state common to GUI & web.

        The conversion of numpy arrays → lists for JSON, colour maps, etc. is
        left to the caller (GUI can consume numpy directly, web needs lists).
        """

        return {
            'snake_positions': self.snake_positions,
            'apple_position': self.apple_position,
            'game_number': self.game_number,
            'score': self.score,
            'steps': self.steps,
            'move_index': self.move_index,
            'total_moves': len(self.moves),
            # These keys may be absent on some subclasses but harmlessly
            # default via getattr – keeps the interface stable across tasks.
            'planned_moves': getattr(self, 'planned_moves', []),
            'llm_response': getattr(self, 'llm_response', None),
            'primary_llm': getattr(self, 'primary_llm', None),
            'secondary_llm': getattr(self, 'secondary_llm', None),
            'paused': self.paused,
            'speed': 1.0 / self.pause_between_moves if self.pause_between_moves > 0 else 1.0,
            'timestamp': getattr(self, 'game_timestamp', None),
            'game_end_reason': getattr(self, 'game_end_reason', None),
            'total_games': getattr(self, 'total_games', None),
        }
```

## BaseGameManager

The whole BaseGameManager class should be used by task0, 1, 2, 3, 4, 5. 

## GameController
BaseGameController is really generic and can be used directly (or at least, almost directly) by task0, 1, 2, 3, 4, 5. At least, Task0 uses it directly (GameController is a subclass of BaseGameController, with inherites everything from BaseGameController and adds no extra functionality).

## RoundManager


## Statistics

