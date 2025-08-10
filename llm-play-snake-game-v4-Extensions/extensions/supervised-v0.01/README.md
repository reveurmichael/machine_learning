### Supervised v0.01 (Extension)

This extension demonstrates a minimal supervised-style agent using the streamlined core API. It does not implement training; it focuses on clean inference and logging.

Key points:
- Inherits `BaseGameManager`; no custom `setup_game` required. `--grid_size` is honored by core.
- Uses public round APIs (`record_round_game_state`) and one-liner save (`save_current_game_json`).
- Factory `create()` returns a simple GREEDY agent for demonstration.
- Streamlit `app.py` serves only as a launcher for `scripts/main.py` (per `final-decision.md`).

Run:
- CLI: `python scripts/main.py --algorithm GREEDY --max-games 10 --grid-size 10`
- Streamlit launcher: `streamlit run app.py`