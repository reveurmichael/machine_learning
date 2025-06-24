# Coordinate System

VITAL: THIS ONE IS VERY IMPORTANT. It's a single-source-of-truth documentation – applies to **all** extensions, all tasks0-5.

IMPORTANT FILE THAT YOU SHOULD NEVER IGNORE.


The entire code-base (core logic, agents, prompts, GUI adapters, and all extensions) adopts **one single right-handed 2-D Cartesian coordinate system**.  All positions are expressed as integer tuples **(x, y)** that respect the following conventions:

* **Origin** : (0, 0) is the **bottom-left** corner of the board.
* **Axes orientation**   
  * **x-axis** grows **to the right**.  
  * **y-axis** grows **upwards**.
* **Board size** : by default 10 × 10. Valid coordinates therefore satisfy 0 ≤ x < 10 and 0 ≤ y < 10.  The same rules extend naturally to different board sizes (see _Scalability_ below).

---
## Direction Keywords <a name="direction-keywords"></a>

Direction strings are *case-sensitive* and always uppercase:

| Direction | Vector Δ(x, y) |
|-----------|----------------|
| `UP`      | ( 0, +1) |
| `DOWN`    | ( 0, –1) |
| `RIGHT`   | (+1,  0) |
| `LEFT`    | (–1,  0) |
| `NONE`    | ( 0,  0) *(initial state only)* |

These vectors are used everywhere: movement logic, prompt generation, replay engine, and GUI adapters.

---
## Visualising the Grid

Below is an ASCII sketch of the default 10 × 10 grid with coordinates labelled.  Each cell shows its **(x, y)** tuple (spaces removed for brevity).  Notice how **y increases upward** while **x increases to the right**.

```
9 | (0,9)(1,9)(2,9)(3,9)(4,9)(5,9)(6,9)(7,9)(8,9)(9,9)
8 | (0,8)(1,8)(2,8)(3,8)(4,8)(5,8)(6,8)(7,8)(8,8)(9,8)
7 | (0,7)(1,7)(2,7)(3,7)(4,7)(5,7)(6,7)(7,7)(8,7)(9,7)
6 | (0,6)...                                             
5 | (0,5)...                                             
4 | (0,4)...                                             
3 | (0,3)...                                             
2 | (0,2)...                                             
1 | (0,1)...                                             
0 | (0,0)(1,0)(2,0)(3,0)(4,0)(5,0)(6,0)(7,0)(8,0)(9,0)
    ----------------------------------------------------
      0    1    2    3    4    5    6    7    8    9    x
```

---
## Worked Examples

### 1  Single-Step Moves

Assume the snake's head is at **(1, 1)** with current direction `NONE` (game start).  All immediate legal moves are:

* `UP`    → (1, 2)
* `DOWN`  → (1, 0)
* `RIGHT` → (2, 1)
* `LEFT`  → (0, 1)

### 2  Path to an Apple

Head = **(4, 4)**, Apple = **(6, 7)**. One safe shortest path could be:

1. `RIGHT` → (5, 4)
2. `RIGHT` → (6, 4)
3. `UP`    → (6, 5)
4. `UP`    → (6, 6)
5. `UP`    → (6, 7) (apple eaten)

Observe that horizontal moves affected **x**, vertical moves affected **y**, exactly matching the direction vectors in the table above.

---
## Consistency Across Components

* **Core logic** (`core/game_logic.py`, `core/game_data.py`, …) performs collision checks and movement updates in this coordinate frame.
* **GUI adapters** translate between this mathematical frame and the pixel-based frame of rendering libraries (e.g., Pygame uses top-left origin, so a flip on the y-axis is performed internally inside `gui/`).
* **Prompts** for LLM agents explicitly describe the coordinate rules to ensure the model reasons in the same frame (see `config/prompt_templates.py`).
* **Extensions** must NOT redefine the coordinate system.  Instead, they reuse the same conventions to maintain interoperability (replays, dashboards, analytics).

---
## Scalability & Custom Boards

Although Task-0 uses a 10 × 10 grid, the code supports arbitrary rectangular sizes.  The semantics remain identical—only the upper bounds change.  

---
## Common Pitfalls

2. **Off-by-one errors** – remember the highest valid index is *size – 1*.
3. **Y-axis confusion** – many graphics toolkits have (0, 0) at top-left; our logic does not.
