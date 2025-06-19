# 🚀 Migration Roadmap: From **v1 (Image-Coordinate)** to **v2 (Cartesian-Coordinate)**

> **Folders**  
> • v1 code-base: `./llm-play-snake-game-v1-image-coordinate/`  
> • v2 code-base: `./llm-play-snake-game-v2-cartesian-coordinate/`

---

## 1. Executive Summary

The transition from **v1** to **v2** rewires the snake-game/LLM agent around a *Cartesian, grid-native* mental model, replacing the *image-pixel* conventions of the first prototype.  
This single (but far-reaching) shift unlocks a cascade of improvements:

1. 🔄 **Bi-directional clarity** between code and prompt: every cell is addressed by `(x, y)` instead of `(row, col)`/pixel.  
2. 🛡 **Robust I/O contract**: LLMs now speak JSON – deterministic for parsers, test-friendly for evaluation.  

The document maps the *motivation*, *philosophy*, *design guidelines*, *challenges*, *solutions* and *lessons* that shaped v2 – so that future iterations start on firmer ground.

---

## 2. Motivation

| Pain-point in v1 | Impact | Goal in v2 |
|------------------|--------|-------------|
| Image-style grid `(row, col)` with **y-down** semantics | • Humans & LLMs intuitively think y-up → cognitive load<br>• Harder to check Manhattan distances<br>• Reversing y on every operation invites bugs | Adopt standard **Cartesian** `(x, y)` with **y-up**; unify math, prompt & rendering |
| Free-form, numbered list of moves | • Parsing brittle (`parse_llm_response` full of regex) <br>• No automatic schema validation | Force **strict JSON**: `{ "moves": [..], "reasoning": "..." }` |

---

## 3. High-Level Design Changes

### 3.1 Coordinate System

v1 (Image):
```python
# v1 config
DIRECTIONS = {
    "UP":    (0, -1),  # negative y
    "RIGHT": (1, 0),
    "DOWN":  (0, 1),   # positive y
    "LEFT":  (-1, 0)
}
```

v2 (Cartesian):
```python
# v2 config
DIRECTIONS = {
    "UP":    (0, 1),   # positive y
    "RIGHT": (1, 0),
    "DOWN":  (0, -1),
    "LEFT":  (-1, 0)
}
```
Key decisions:

1. Origin `(0,0)` is **bottom-left**, matching textbooks and most grid RL environments.
2. Board storage (`self.board[y, x]`) remains row-major but helper methods convert automatically; `_verify_coordinate_system()` sanity-checks this at start-up.
3. Movement math moves to `head_x + dx`, `head_y + dy` for legibility.

### 3.2 Prompt Contract

* **v1** – Free text → numbered list.  
* **v2** – Strict JSON, with placeholders auto-filled by the engine:

```jsonc
{
  "moves": ["UP", "RIGHT", "RIGHT", "UP"],
  "reasoning": "Keeps safe distance from tail while closing Manhattan gap."
}
```

Benefits:

* Deserialisation is one line: `data = json.loads(response)` after cleaning.  
* Unit tests can assert `validate_json_format(data) is True`.  
* Other languages / dashboards can consume results without regex.

### 3.3 Utility Module `json_utils.py`

Centralises:

* Extraction from raw text, fenced code blocks, or malformed LLM output.
* Pre-processing (single-quotes → double, trailing commas, etc.).
* Schema validation.

This halves duplicate regex code across `snake_game.py` and `llm_client.py`.

## 4. Development Philosophy

3. **LLM friendliness** – The engine *speaks the LLM's language*: structured examples, consistent coordinate system, rich context yet bounded token usage.  

---

## 5. Migration Guidelines (for contributors)

1. **Keep prompts self-contained** – they should explain the entire game without requiring outside docs.  

---

## 6. Key Challenges & Solutions

| Challenge | Why It Hurt | v2 Solution |
|-----------|-------------|-------------|
| **Coordinate flip-flops** between UI, logic, prompt | Ghost bugs: snake visually at (5,1) but prompt says (1,5) | Single source of truth: Cartesian in logic, translation at UI draw only |
| LLM sometimes returns stray prose before JSON | JSON decoder fails → game stuck | `extract_valid_json()` tolerates markdown, commentary blocks |

---

## 8. Lessons for Programmers & AI Practitioners

1. **Representation matters** – A well-chosen coordinate system simplifies both *human* and *machine* reasoning.  
2. **LLM contracts should be machine-readable** – JSON > prose. Design for parse-ability first, eloquence second.  
4. **Logs are lineage** – Keep every prompt/response; tomorrow's bug fix depends on yesterday's trace.  
5. **Treat LLMs as unreliable agents** – Always validate and sanitise their output before acting.  
6. **Incremental refactors beat rewrites** – v2 re-organises without throwing away v1; we can A/B quickly.  
7. **Edge-case unit tests** guard coordinate changes – When semantics flip (y-up vs y-down) tests catch regressions instantly.


---

## 9. Concluding Thoughts

The journey from **v1** to **v2** underscores a universal engineering theme: *small abstractions drive big leverage*. By simply aligning the game's mental model (*Cartesian grid*) with the LLM's reasoning abilities and enforcing a strict *structured I/O contract*, we unlocked reliability, maintainability, and new research horizons – all with minimal churn to existing GUI/gameplay code.

Keep these principles in mind as you branch into future versions; they will repay themselves in stability and developer happiness.

> "*Abstraction is not about hiding the truth, it's about choosing which truth to optimise for."* – this roadmap is our compass.


