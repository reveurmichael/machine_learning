# Extensions v0.04: The Language Generation Phase (Heuristics Only)

## 🎯 **Core Philosophy: Teaching an LLM to "Think" Like a Heuristic**

The `v0.04` extension is a unique and highly specialized version that applies **exclusively to the `heuristics` extension**. There is no `v0.04` for any other algorithm type.

Its sole purpose is to act as a **language-rich data producer** for the purpose of fine-tuning Large Language Models (LLMs). It answers the question:

> "Can we teach an LLM to replicate the logical, step-by-step reasoning of a classic heuristic algorithm?"

To do this, `heuristics-v0.04` modifies the heuristic agents from `v0.03` to not only calculate the next best move but also to **articulate in natural language *why* that move was chosen**.

## 🏗️ **Architectural Approach: A Minor Evolution of v0.03**

The `v0.04` extension is not a major architectural leap. It is an enhancement of the existing `v0.03` structure.

*   **Foundation:** It inherits the full structure of `heuristics-v0.03`, including the `app.py`, and `scripts/` directories.
*   **Agent Modification:** The core change happens within the agent classes (e.g., `BFSAgent`, `AStarAgent`). These agents are modified to generate a human-readable "reasoning" string at each decision point.



## 📜 **The Mandatory Data Format: JSONL for Fine-Tuning**

To create effective training data for LLMs, the output of `heuristics-v0.04` **must** be in the **JSONL (JSON Lines)** format. Each line in the output file represents a single training example, structured as a JSON object.

### **JSONL Schema**

Each JSON object must contain two keys: `"prompt"` and `"completion"`.

*   **`"prompt"`**: A textual description of the game state. This serves as the input to the LLM.
*   **`"completion"`**: A textual description of the agent's chosen move and the reasoning behind it. This is the target output the LLM will be trained to generate.

This format is specifically designed to be easily consumed by standard LLM fine-tuning pipelines (e.g., for Supervised Fine-Tuning). All file operations must use UTF-8 encoding for cross-platform compatibility (SUPREME_RULE NO.7).

## 📋 **Compliance Checklist: The Definition of Done**

The `heuristics-v0.04` extension is considered complete and successful if:

- [ ] Does it build directly upon the `heuristics-v0.03` structure?
- [ ] Are the agent classes modified to generate natural language explanations for their moves?
- [ ] Does this script produce a `.jsonl` file where each line is a valid JSON object?
- [ ] Does each JSON object strictly adhere to the `{"prompt": "...", "completion": "..."}` schema?
- [ ] Are all file operations using UTF-8 encoding (SUPREME_RULE NO.7)?

---

> **The `heuristics-v0.04` extension is a critical bridge between symbolic AI and neural AI. Its success is measured by its ability to produce clean, structured, and linguistically rich datasets for training the next generation of agents.**
