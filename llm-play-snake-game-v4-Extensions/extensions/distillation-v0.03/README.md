# Distillation v0.03 – Knowledge Distillation with Evaluation Dashboard

> Second-citizen extension – **stand-alone** when paired with `extensions/common/`.

---

## 🌟 What's new in v0.03?

1. **Evaluation suite** – perplexity, exact-match accuracy, and distilled-model size reduction metrics.
2. **Streamlit dashboard** (`app.py`) – run distillation, monitor loss curves in real-time, inspect teacher ↔ student divergences.
3. **CLI parity** (`cli.py`) – `train` & `evaluate` sub-commands with LoRA / QLoRA knobs.
4. **Rich docs** (this file!) – quick-start, API reference, architecture diagram.

v0.03 **re-uses 100 % of the training pipeline** introduced in v0.02; no breaking changes.

---

## 🏗️ Folder layout

```
extensions/distillation-v0.03/
├── __init__.py          # DistilConfig, public API
├── pipeline.py          # DistillationPipeline (imported from v0.02)
├── evaluation.py        # 🔄 NEW – EvaluationSuite
├── cli.py               # 🔄 NEW – Click CLI (train / evaluate)
├── app.py               # 🔄 NEW – Streamlit dashboard
└── README.md            # ← you are here
```

---

## 🚀 Quick start

### 1 . Install extras
```bash
pip install -r extensions/common/requirements-llm.txt  # HF & LoRA deps
```

### 2 . Train distilled model
```bash
python -m extensions.distillation-v0.03.cli train \
  --teacher gpt2-xl \
  --student distil-gpt2 \
  --dataset-path ./logs/extensions/datasets/grid-size-10/*.jsonl \
  --epochs 3 --batch-size 8 --use-lora
```

### 3 . Evaluate
```bash
python -m extensions.distillation-v0.03.cli evaluate \
  --model-path ./logs/extensions/models/grid-size-10/distil-gpt2_20250623_105012 \
  --dataset-path ./logs/extensions/datasets/grid-size-10/test.jsonl
```

### 4 . GUI dashboard (optional)
```bash
streamlit run extensions/distillation-v0.03/app.py
```

---

## ⚙️ Key classes

| File | Class | Purpose |
|------|-------|---------|
| `__init__.py` | `DistilConfig` | Immutable config dataclass shared by pipeline & CLI. |
| `pipeline.py` | `DistillationPipeline` | Orchestrates teacher-student forward pass, computes Hinton loss, handles checkpointing. |
| `evaluation.py` | `EvaluationSuite` | Loads a model and computes perplexity / accuracy on held-out data. |
| `cli.py` | `@click.group()` | Thin wrapper exposing `train` and `evaluate` commands. |
| `app.py` | Streamlit | Friendly UI for launching runs & visualising metrics. |

All heavy lifting happens inside `DistillationPipeline` – v0.03 merely **extends** it with evaluation and UI wrappers (Template-Method + Decorator patterns).

---

## 📝 Design patterns in action

1. **Singleton** – internal `FileManager` from `extensions/common/` guarantees one log path per run.
2. **Factory** – `create_model()` builds teacher/student; supports HF, QLoRA & custom ckpts.
3. **Strategy** – interchangeable `LossStrategy` (Hinton, cosine, KL).  Default = Hinton.
4. **Template-Method** – `DistillationPipeline.run()` orchestrates high-level flow; hooks overridden by dashboard for live plots.
5. **Observer** – metrics reporter notifies CLI or Streamlit sinks in real-time.

All patterns are accompanied by **verbose docstrings** for educational value.

---

## 🔒 Output directories

* Models → `logs/extensions/models/grid-size-N/distillation-v0.03_{timestamp}/`
* TensorBoard & CSV logs → sub-dir `training_logs/` inside the model folder.

The folder conventions honour the global guidelines in `docs/extensions-guideline/`.

---

## ❓ FAQ

**Q: Why JSONL datasets?**  Because LLMs operate on tokens; JSONL is the lingua franca for HF datasets.

**Q: Does v0.03 break v0.02?**  No – v0.03 *wraps* v0.02.  Old scripts continue to work.

**Q: Can I plug in my own loss?**  Yes – subclass `LossStrategy` and pass `--loss-strategy my_module:MyLoss` to the CLI.

---

## 🛠️ Roadmap

| Version | Focus |
|---------|-------|
| v0.04 | Better eval (BLEU for commentary), HuggingFace Spaces demo |
| v0.05 | Knowledge distillation from **Task-0 live prompts** (online KD) |


---

*Last updated 2025-06-23* 