### Do you need multiple data formats (CSV, NPZ, Parquet) for dataset preparation?

**Short answer:**

* For many classic supervised learning algorithms (XGBoost, LightGBM, Catboost, simple neural networks), **CSV (tabular) format is often sufficient**.
* However, for deep learning models dealing with **sequential data (RNNs, LSTMs)** or **graph-structured data (GNNs)**, specialized formats like **NPZ (NumPy arrays)** or **Parquet (efficient columnar storage)** provide significant benefits.

---

### Why might CSV be enough in many cases?

* **XGBoost / LightGBM / Catboost/ Random Forest:** These typically require flat tabular data, which CSV handles naturally.
* **Basic feedforward neural networks (MLPs):** Also prefer flattened vector inputs, easily stored as CSV rows.
* **Evolutionary algorithms:** Usually operate on fixed-size feature vectors or parameter sets, again CSV is straightforward.

So, if your supervised learning tasks primarily use tree-based methods or MLPs and your evolutionary algorithms don’t require complex structured inputs, CSV alone can be a **simpler and sufficient choice**.

---

### When might you want NPZ or Parquet?

* **Sequential models (RNN, LSTM, GRU):**
  Training these models requires time-series or sequence data, where each sample is a sequence of states or features over time. Storing this as CSV is possible but cumbersome and inefficient.
  **NPZ (NumPy binary arrays)** is preferred as it preserves array shapes and allows fast I/O.

* **Graph Neural Networks (GNNs):**
  GNNs need data representing nodes, edges, and graph connectivity. CSV is not well-suited to this inherently relational data.
  **Parquet** or specialized graph data formats are much better for efficiently storing and loading large graph datasets.

* **Large-scale datasets or complex features:**
  Parquet files support efficient compression, columnar access, and partial reads, useful for big data or heterogeneous features.

---

### Summary recommendation:

| Use Case                                 | Recommended Format         | Reason                                          |
| ---------------------------------------- | -------------------------- | ----------------------------------------------- |
| Classic supervised learning (trees, MLP) | CSV                        | Simple, compatible, easy to debug               |
| Sequential models (RNN, LSTM)            | NPZ                        | Preserves sequence shape, efficient I/O         |
| Graph models (GNN)                       | Parquet (or graph formats) | Supports complex relational data and efficiency |
| Evolutionary algorithms                  | CSV (usually)              | Fixed-length vector inputs, simple format       |

---

### Final notes:

* You **can start with CSV for most models and evolutionary algorithms** to reduce complexity.
* If you plan to support **deep sequence or graph models seriously**, having NPZ and Parquet support **is recommended** for scalability and ease of training.
* Keeping the dataset generation modular, allowing different output formats from the same raw gameplay data, is a good engineering practice.
* This flexibility also future-proofs your framework for evolving model architectures.

---

**In conclusion:**
CSV files are enough for many supervised learning and evolutionary algorithm training pipelines — especially at the start. But for advanced models like RNNs or GNNs, supporting NPZ and Parquet is highly advisable.

## Our decision
Yes, we will go for CSV, NPZ and Parquet, three of them. They will go into the same folder.
