### Supervised v0.01 (Extension)

This extension demonstrates a minimal supervised-style agent using the streamlined core API. It focuses on clean inference and logging.

Algorithms (factory `create()`):
- MLP, CNN, RNN, LSTM, LIGHTGBM, XGBOOST (each optionally accepts `model_path`)

Key points:
- Inherits `BaseGameManager`; no custom `setup_game` required. `--grid_size` is honored by core.
- Uses public round APIs (`record_round_game_state`) and one-liner save (`save_current_game_json`).
- Streamlit `app.py` serves only as a launcher for `scripts/main.py` (per `final-decision.md`).

Run:
- CLI: `python scripts/main.py --algorithm MLP --max-games 10 --grid-size 10 [--model-path /path/to/model.pkl]`
- Streamlit launcher: `streamlit run app.py`