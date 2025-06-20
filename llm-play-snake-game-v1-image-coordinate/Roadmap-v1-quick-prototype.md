# Quick-and-Dirty Roadmap — LLM Plays Snake (v1 prototype)

> **This file is intentionally rough.**  We care more about seeing something work than about polished code or perfect architecture at this stage.

---

## 0. Starting Point

* **Base repo:** [`SnakeQ`](https://github.com/hexontos/SnakeQ)  – we cherry-pick just enough to get a working Snake game with PyGame rendering.
* **What we keep:** grid logic, simple GUI, event loop.
* **What we drop:** Deep-Q-Network training, replay buffers, anything TensorFlow.

---

## 1. What We're Trying To Prove

1. An LLM can read a **text description** of the board and spit back a **reasonable move** (`UP / DOWN / LEFT / RIGHT`).
2. We can feed that move into the existing Snake loop and the snake reacts.
3. The loop runs for at least a few apples before crashing for some LLM models (e.g. deepseek-reasoner) – that's good enough for now.

If those three things happen, we call **v1 "good enough."**  Bugs & messy code can wait.

---

## 2. Minimal Task List (checkboxes = aspiration, not rigorous)

- [ ] Strip out RL training references from SnakeQ.
- [ ] Refactor snake positions to `[row, col]` (image-coordinate mindset).
- [ ] Add **LLMClient** wrapper with a single method: `generate_response(prompt)`.
- [ ] Hard-code one provider first (e.g. `ollama` local model) to keep deps light.
- [ ] Create very naïve `PROMPT_TEMPLATE` that pastes board ASCII + asks for next move.
- [ ] Parse the LLM output with a dumb regex – if anything illegal, default to `RIGHT`.
- [ ] Log every prompt & response to folder (plain text is fine).

---

## 3. Milestone Sketch (extremely loose)

| Day | Goal | Notes |
|-----|------|-------|
| **D0** | Game loop runs minus RL bits | "Hello Snake" window opens, manual arrow keys work. |
| **D1** | LLM client returns *some* answer | Could even be hard-coded for first run. |
| **D2** | First automated move executed | Snake moves on its own once. |
| **D3** | Stream of moves until first crash | If snake eats 1 apple – celebrate 🎉 |
| **D4** | Basic logs + quick README  | Enough to tweet / share internally. |

If we slip by a few days – no problem.  This is exploratory.

---

## 4. Known Limitations We'll Ignore (for now)

* **Prompt bloat** – We'll probably send too many tokens; optimisation later.
* **Hard-coded constants** – Grid size, tick rate, etc. live in `config.py` without nice CLI flags.
* **No tests** – I tested the code manually. Afterall, we are just trying to prove the concept.

---

## 5. Quick Glossary

| Term | Meaning in This Repo |
|------|----------------------|
| **Image Coordinates** | `(0,0)` is top-left, `row` increases downward, `col` increases rightward. |
| **LLM** | Any model that answers text in = text out (ChatGPT, Ollama, Mistral, etc.). |
| **Move** | One of `UP`, `DOWN`, `LEFT`, `RIGHT`.  If invalid we default to `RIGHT`. |

---

Feel free to _hack_, commit, and push.  We'll clean up a little bit in v2 and make it more robust in v3.








