# ❌ **Potential subtle SSOT risks or violations**

## 1️⃣ Explanation regex replacements may be partial or brittle

### Issue

In `_update_explanation_with_ssot_metrics`, you rely on regex patterns to replace position and metric values in the explanation text. However, if the original explanation text includes variations (e.g., "apple position is at (X, Y)", or "Manhattan dist: X"), or uses different spacing or formatting, those won't be caught.

As a result, some stale numeric strings could remain and conflict with SSOT metrics.

### Suggestion

* **Stronger approach**: Instead of "fixing" the explanation string via regex, regenerate or re-render it directly from the SSOT metrics.
* Alternatively, keep regex but also perform a final **strict numeric check**: scan for any number tokens matching old metric values and fail if found.

---

## 2️⃣ CSV feature `target_move` uses `move` directly

This is okay in most cases, but if `move` was initially inconsistent with `game_state`, you might propagate that inconsistency into the CSV file.

### Suggestion

Validate `move` against `valid_moves` recomputed from `game_state`. If `move` is not valid, fail fast or fix.

---

## 3️⃣ `snake_positions` membership checks in `_format_prompt`

In `_format_prompt`, when determining `valid_moves`, you do:

```python
if (0 <= next_pos[0] < grid_size and
    0 <= next_pos[1] < grid_size and
    next_pos not in snake_positions):
```

But `snake_positions` is a list of lists; `next_pos` is a tuple. So `next_pos not in snake_positions` always evaluates True.

This could cause `valid_moves` to incorrectly include moves that would result in collisions.

### Suggestion

Convert `snake_positions` to set of tuples first:

```python
snake_body_set = set(tuple(p) for p in snake_positions)
```

Then check `next_pos not in snake_body_set`.

---

## 4️⃣ Possible reuse of stale metrics in explanations

In `_extract_jsonl_record`, you flatten the explanation and then update it. But if the original `explanation` dict contains "metrics" keys, you partially drop them. However, if any nested subtext or cached textual snippet (from an earlier model output) is stored inside, some metric numbers might remain.
