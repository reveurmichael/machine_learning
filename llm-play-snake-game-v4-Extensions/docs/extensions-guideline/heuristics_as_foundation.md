# Heuristics – The Bedrock of the ML Road-Map

> *Task-1 may look "classical", but it is the heartbeat of every downstream
>  experiment.*

## 1. Why Heuristics come first

| Role | Explanation |
|------|-------------|
| **Ground-truth generator** | BFS, A*, Hamiltonian etc. create *perfect* move labels that power Supervised Learning (Task-2). |
| **Curriculum designer** | RL uses heuristic roll-outs as warm-start or curriculum targets. |
| **Language tutor** | v0.04 converts heuristic reasoning into natural-language JSONL – the seed corpus for LLM fine-tuning (Task-4). |
| **Safety benchmark** | Even after learning, new agents are compared against heuristic survival stats. |

## 2. Data lineage

```
heuristics logs
    ├─ CSV (v0.03)  ──┐
    │                │
    │   supervised train  (Task-2)
    │                │
    └─ JSONL (v0.04) ──┤
                     fine-tune LLM  (Task-4)
```

Breaking the CSV or JSONL schema will ripple into **three** other tasks – keep
backwards compatibility inside the `logs/extensions/datasets/grid-size-N/`
hierarchy.

## 3. Coding guidelines for heuristic extensions

1. **Never assume 10×10** – detect `grid_size` and save to the correct folder.  
2. **Document the algorithm** – future researchers will debug from your
   docstrings.
3. **Save metadata** – include algorithm name, parameters, and git‐SHA in
   `summary.json`.
4. **Use common helpers** – `DatasetDirectoryManager`, `GridSizeDetector` to
   avoid duplicated bugs.

## 4. Supervised-learning's obligation

Supervised packages **must acknowledge** in their module docstrings that their
only data source is *heuristics*.  Any feature-engineering tweak should be
reflected back into the heuristic → CSV converter so the contract remains
synchronised.

---

*Last updated : 2025-06-23* 