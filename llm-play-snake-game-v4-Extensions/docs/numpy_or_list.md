
• Runtime logic (movement, collision, AI) already relies on NumPy arrays everywhere the math actually happens—`GameController`, `check_collision`, etc.  
• Persistent artefacts (JSON logs) must remain plain lists because NumPy arrays aren’t JSON serialisable.  The replay layer therefore reads lists straight from disk and only needs them in list form to step through moves or to feed them to the front-end.

Replay code you pointed at:

```python
# replay_engine.py
self.apple_positions: List[List[int]] = []       # pure JSON payload
...
# replay_utils.py
apples = [[pos["x"], pos["y"]] for pos in ...]   # still JSON payload
```

Those structures never call collision maths or vector arithmetic—they’re just replay data and are later pushed to the browser, so there’s no real benefit in converting them to NumPy.

Therefore:

1. Keep NumPy arrays for in-memory, real-time game computation (GameController, check_collision, AI logic).  
2. Keep Python lists for long-term storage and for anything that’s marshalled to JSON (logs, replay files, web APIs).  
3. Perform cheap on-demand conversions (`arr.tolist()` when serialising, `np.array(list)` only if you ever need to do maths on replay data).

This gives clear separation:

```
| Layer            | Position type |
|------------------|---------------|
| Core gameplay    | np.ndarray    |
| Logs / JSON      | list[int]     |
| Replay front-end | list[int]     |
```

Migrating the replay modules to NumPy would touch many files, add conversions back to lists for JSON, and deliver no tangible performance or clarity benefit.  So the lowest-impact, easiest-to-maintain path is to leave them as is and document the convention.

