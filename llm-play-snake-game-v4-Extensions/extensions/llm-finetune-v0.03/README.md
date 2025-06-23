# LLM Fine-Tune v0.03 – Interactive Dashboard + Evaluation

v0.03 demonstrates a *natural software evolution* from CLI-only tooling to a
fully-featured **Streamlit UI** – without adding backward-compat baggage.
It simply *wraps* the battle-tested v0.02 pipeline and sprinkles UX sugar on
 top.

## 🆕 What's New in v0.03?

| Area             | v0.02 | v0.03 |
|------------------|-------|-------|
| Multi-dataset    | ✅    | ✅    |
| LoRA/QLoRA       | ✅    | ✅    |
| Streamlit UI     | ❌    | **✅** |
| Quick evaluation | ❌    | **✅** |
| Code duplication | –     | **0 LOC** re-used pipeline |

### 1. **Streamlit `app.py`**
* Drag-and-drop JSONL datasets.
* Slider widgets for epochs, batch size, LoRA rank.
* Generates the exact CLI command for reproducibility.
* One-click perplexity calculator after training.

### 2. **`EvaluationSuite`**
* Tiny helper (≈40 loc) – loads model & dataset, returns perplexity.
* Designed to grow with BLEU / exact-match metrics later.

### 3. **Unified CLI**
```
python -m extensions.llm_finetune_v0_03.cli train   …   # training
python -m extensions.llm_finetune_v0_03.cli evaluate …   # quick PPL
```

## 🏗️ Architecture
```
llm-finetune-v0.03/
├── __init__.py      # re-exports pipeline + EvaluationSuite
├── cli.py           # training / evaluation commands
├── app.py           # Streamlit UI (optional)
└── (no duplicate pipeline code!)
```

The **single source of truth** for training remains `FineTunePipeline` in
v0.02.  v0.03 only adds presentation & analytics layers.

## 📜 Philosophy Compliance
* **No backward-compat language** – purely forward-looking.
* **DRY** – zero duplication.
* **Educational** – shows how to evolve CLI tools into interactive apps. 