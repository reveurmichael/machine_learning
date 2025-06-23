## 🧩 General Overview

**v0.04** focuses exclusively on **dataset generation with natural language explanations**, specifically for the **heuristics** family of Snake game agents.

The core feature of v0.04 is the ability to generate **CSV datasets** that include both:

* **Structured decision data** (e.g., board state, direction, position),
* and **rich natural language descriptions** explaining **why each decision is made**.

This dataset is intended for **fine-tuning large language models (LLMs)** in Task 4, enabling them to learn and imitate heuristic algorithms through language-grounded reasoning.

---

## 🤖 Heuristics v0.04

v0.04 applies **only to heuristics**.
There is **no v0.04** planned for:

* Supervised learning models
* Genetic or evolutionary algorithms
* reinforcement learning models

### 🔁 Built on Top of v0.03

Since v0.04 extends v0.03, it inherits the web interface, replay system, and dataset export infrastructure. However, v0.04 introduces a crucial innovation:

> At **each key decision point** in the heuristic logic, the agent will **generate detailed natural language explanations** describing the reasoning behind its action.

These explanations will be logged alongside standard metadata (e.g., grid state, next move) in CSV format.

### 🧠 LLM Fine-Tuning Use Case (Task 4)

These **language-rich CSV datasets** are designed specifically to:

* Teach LLMs how to play the Snake game
* Ground decision-making in explicit, interpretable reasoning
* Provide a training corpus for supervised fine-tuning (e.g., SFT or RFT) of open-weight models

Example data entries will include columns like:

| step | head\_pos | apple\_pos | move  | reasoning\_sentence                                                                    |
| ---- | --------- | ---------- | ----- | -------------------------------------------------------------------------------------- |
| 14   | (3,4)     | (6,7)      | RIGHT | “The apple is to the upper-right, and RIGHT leads me closer while avoiding obstacles.” |

This forms the bridge from classic symbolic reasoning to neural policy learning.

---

## 📦 Supervised Learning Models

There is **no v0.04 for supervised learning models**.

The existing supervised extensions (v0.01–v0.03) already:

* Train on datasets exported from heuristics v0.03
* Cover a wide variety of models (XGBoost, LightGBM, PyTorch NN/CNN/RNN/GNN)
* Include training, evaluation, and visualization components

These models do **not require** language-annotated data at this time.

---

## ✅ Summary

| Category           | v0.04 Support | Notes                                                               |
| ------------------ | ------------- | ------------------------------------------------------------------- |
| Heuristics         | ✅ Yes         | CSV + language reasoning for LLM fine-tuning                        |
| Supervised Models  | ❌ No          | v0.03 is sufficient for training + evaluation                       |
| Genetic Algorithms | ❌ No          | v0.03 remains the latest; not involved in language generation tasks |

**v0.04 = Dataset for LLM fine-tuning with rich explanations.**
