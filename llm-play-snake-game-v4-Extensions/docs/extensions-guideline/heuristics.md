## 🧭 Common Heuristics for Snake Game

### 1. **Greedy Heuristics**

Choose the move that brings the snake **closest to the apple**.

* **Manhattan Distance Heuristic**

  $$
  h(x, y) = |x - x_{apple}| + |y - y_{apple}|
  $$

  * Fast and simple.
  * Fails when path is blocked by the body.

* **Euclidean Distance (less common in grid games)**

  $$
  h(x, y) = \sqrt{(x - x_{apple})^2 + (y - y_{apple})^2}
  $$

  * Smooth but not optimal for grid-based movement.

✅ **Pros**: Simple, efficient
❌ **Cons**: Short-sighted — can trap the snake in tight spaces.

---

### 2. **Safety-First Heuristics**

Prioritize **survival** over score.

* **Flood Fill Heuristic (Space Availability)**
  Estimate how much free space a move leads to using BFS/DFS. Choose the move that leads to **largest available region**.

* **Longest Path Heuristic**
  Try to follow the longest path to the apple to delay reaching it — keeps tail far from head and prevents self-collision.

* **Tail-Aware Heuristic**
  Ensure that the snake **can reach its own tail** after taking a step (i.e., simulate that the tail will vacate that cell).

✅ Pros: Excellent survival behavior
❌ Cons: Slower, more complex to compute

---

### 3. **Lookahead-Based Heuristics**

Use **limited-depth simulations** to evaluate future consequences.

* **Minimax with Heuristics**
  Evaluate next `N` steps and pick the one with best final outcome score.

* **Monte Carlo Rollouts**
  Play random simulations from current move and choose the move that leads to most survivable runs.

* **A*-like Search*\*
  Use `f(n) = g(n) + h(n)` where:

  * `g(n)` is path cost so far
  * `h(n)` is estimated distance to apple
    Use flood-fill to ensure solution path exists.

✅ Pros: Smarter behavior
❌ Cons: Slower than greedy

---

### 4. **Rule-Based / Priority Heuristics**

Use manually designed logic:

```python
if apple in line_of_sight and path is safe:
    go to apple
else:
    follow wall
```

Variants:

* Always go clockwise
* Always turn left if path is blocked
* Stick to the perimeter when near full capacity

✅ Good for baselines
❌ Brittle and hard to scale

---

### 5. **Cycle-Based Heuristics**

Use a **precomputed Hamiltonian Cycle** or simple loop around the grid.

* Guarantees survival by following cycle
* Optional: add shortcut logic when safe

✅ Theoretically infinite survival
❌ Poor apple efficiency unless optimized

---

### 6. **Potential Field Heuristics**

Assign a **potential value to each cell**:

* Positive near apple
* Negative near walls/snake body
* Snake moves toward gradient

Like electromagnetism or heat maps.

✅ Smooth control
❌ Prone to local minima (can get stuck)

---

## 🧠 Hybrid Heuristics

You can also **combine** the above ideas. For example:

```python
score = -manhattan_distance + 2 * flood_fill_area - 5 * danger_risk
```

Or use:

* **Greedy path to apple if it’s safe**, otherwise
* **Flood fill to choose largest region**

---

## ✨ Extras

| Heuristic Name      | Description                                           | Use case             |
| ------------------- | ----------------------------------------------------- | -------------------- |
| Apple-on-Tail Check | Only move toward apple if tail is reachable           | Prevents growth trap |
| Dead-End Avoidance  | Avoid paths leading to no exit                        | Survival             |
| Cycle Rejoin Heur.  | Shortcut to apple only if it re-enters cycle properly | Hamiltonian+         |

---

## Want to Try One?

If you want, I can help you **code a heuristic agent** based on:

* Flood fill
* Apple-distance
* Tail-following
* Longest-safe-path

Let me know your use case (classroom, RL baseline, LLM feedback, etc.) and I can tailor one!
