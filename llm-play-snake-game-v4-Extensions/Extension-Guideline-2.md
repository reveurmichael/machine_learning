I am currently considering using the **heuristics-v0.03** extension to generate datasets for training various supervised learning models such as **XGBoost, LightGBM, PyTorch-based neural networks (including CNN, RNN), and graph neural networks (PyTorch Geometric)**.

### Key Questions to Address:

* **DATA\_FORMAT:** What is the best format for storing the dataset? Should it be CSV, NPZ, Parquet, or something else?
* **DATA\_STRUCTURE:** How should the dataset be structured and encoded for optimal use by different model types?

The function that converts `game_N.json` files into the desired `DATA_FORMAT.DATA_STRUCTURE` will be implemented in v0.03 (not in v0.01 or v0.02). Since this functionality will be useful across multiple extensions, it makes sense to place it in a **common utilities folder** for easy reuse.

When training supervised learning models, a command-line option like `--dataset-path` will allow loading one or multiple dataset directories (e.g., `--dataset-path dir1 dir2 dir3`) from `./ROOT/logs/extensions/`, supporting variable-length input paths.

---

### Supervised Learning Models Versioning:

We will follow a similar versioning approach for supervised learning extensions, analogous to heuristics:

* **v0.01:** Focused on neural networks using PyTorch (basic training pipeline).
* **v0.02:** Supports a broad range of supervised models — XGBoost, LightGBM, catboost, neural networks, CNN, RNN, GNN, etc. — but **without GUI, replay, or web interface**.
* **v0.03:** Adds full user interfaces including GUI, replay features, web and Pygame modes, and a Streamlit `app.py` for interactive training and evaluation.

Currently, the plan is to develop only **v0.01 and v0.02** for supervised models.

---

### Heuristics v0.04 Considerations:

v0.04 builds upon v0.03 and focuses on **generating rich, long-form natural language explanations at every key decision point** within the heuristics. This is crucial for **Task 4: fine-tuning large language models**. By embedding detailed reasoning in the dataset, we aim to teach LLMs how heuristics work in the Snake game through explicit natural language guidance.

---

### Genetic/Evolutionary Algorithms:

We will also implement genetic/evolutionary algorithm extensions with versions v0.01, v0.02, and v0.03. However, these will **not generate language explanations** like heuristics v0.04.

Their structure and functionality will be similar to heuristics v0.01–v0.03. Detailed documentation and design are already included in the respective markdown files:

* `extensions-v0.01.md`
* `extensions-v0.02.md`
* `extensions-v0.03.md`
