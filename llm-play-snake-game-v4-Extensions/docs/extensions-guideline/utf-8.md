# UTF-8 Everywhere – Practical Guide for Snake-Game Extensions

> **Scope:** Heuristic / RL / Supervised extensions (v0.01 → v0.04) and any
> future tooling that spawns subprocesses or writes log / dataset files.
>
> **Audience:** Extension developers & contributors.

---

## 1  Why UTF-8 as the Single Source of Truth?

*   Cross-platform ⚙️    – Windows 默认仍可能使用 **GBK** / **CP-1252** 等 legacy code pages。
*   Emoji & CJK Support 🎉    – 项目日志里大量使用 ✅ ⚠️ 📂 等字符。
*   Consistency 📦    – 全部数据集（CSV / JSONL）与日志文件需可由任何工具链解析。

> **Rule of thumb:** *Any bytes that leave your Python process MUST be encoded
> in UTF-8, unless the target API explicitly dictates otherwise.*

---

## 2  Printing Safely to Console

### 2.1  Detection & Fallback

`utils/print_utils.py` performs:

```py
_ENCODING_IS_UTF = bool(sys.stdout.encoding and 'utf' in sys.stdout.encoding.lower())
```

* If console encoding is **UTF-8**, it prints emoji/color normally.
* Otherwise (e.g. Windows + GBK), `_safe_text()` replaces un-encodable bytes so
  you get legible output instead of `UnicodeEncodeError`.

### 2.2  Your Responsibility

* **Always** print via the helper functions (`print_info`, `print_warning`, …).
* **Never** call `print("✅ …")` 直接输出 emoji。

---

## 3  Writing Files

```py
# Always specify encoding!
with open(path, "w", encoding="utf-8") as fh:
    fh.write(text)
```

* `json.dump(..., ensure_ascii=False)` keeps non-ASCII characters intact.
* `csv` – pass `encoding="utf-8"` when opening file handles.

---

## 4  Subprocess Handling (windows friendly)

### 4.1  Environment Variable: `PYTHONIOENCODING`

In `extensions/common/utils/dataset_game_runner.py` we enforce:

```py
env.setdefault("PYTHONIOENCODING", "utf-8")
```

* Guarantees that **child Python** processes write UTF-8 to `stdout`/`stderr`.

### 4.2  Binary Capture → Manual Decode

```py
result = subprocess.run(..., text=False, capture_output=True)
stdout = result.stdout.decode("utf-8", errors="replace")
```

* Avoids CPython's internal `readerthread` using system locale.

---

## 5  Common Pitfalls & Remedies

| Symptom                                      | Fix                                               |
|----------------------------------------------|----------------------------------------------------|
| `UnicodeEncodeError: 'gbk' codec can't …`     | Use print helpers OR set `PYTHONIOENCODING`        |
| `UnicodeDecodeError` when reading child logs | Capture as bytes → `.decode('utf-8', errors='replace')` |
| Mojibake in CSV                              | Always open file with `encoding='utf-8'`           |

---

## 6  Checklist for New Code

1.  [ ] All `open()` calls specify `encoding="utf-8"`.
2.  [ ] Console output routed through `utils.print_utils`.
3.  [ ] New subprocesses inherit `PYTHONIOENCODING=utf-8` *or* capture bytes.
4.  [ ] Unit tests pass on **Windows + PowerShell (GBK)** and **Linux + UTF-8**.

---

> **Remember:** UTF-8 is *the* lingua franca. One minute of encoding diligence
> saves hours of cross-platform debugging later. 🚀 