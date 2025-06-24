# Final Decision 0

## 📚 Purpose

This short meta-document explains how the **Final Decision Series (N = 0-10)** fit together and serves as an index for quick navigation.  Each numbered decision is *self-contained* yet **coherent** with the others; together they form the single source of truth for all architectural and implementation standards.

## 🔢 Document Map

| Doc | Theme | Key Topics |
|-----|-------|-----------|
| **Final-Decision-0** | *You are here* | Meta-guidelines & index |
| **Final-Decision-1** | Logs & Directory Structure | Data/Model paths, grid-size hierarchy |
| **Final-Decision-2** | Config & Validation Standards | Config folders, validation system, singleton overview |
| **Final-Decision-3** | Singleton Pattern Details | Uses existing `SingletonABCMeta` from `utils/singleton_utils.py` |
| **Final-Decision-4** | Agent Naming Conventions | `agent_*.py`, `*Agent` classes, validation checklist |
| **Final-Decision-5** | Extension Directory Templates | v0.01 → v0.03 evolution pattern |
| **Final-Decision-6** | Path Management | Mandatory use of `extensions/common/path_utils.py` |
| **Final-Decision-7** | Factory Pattern Extensions | Additional factories & design philosophy |
| **Final-Decision-8** | Factory Implementation Details | Layered architecture, phased rollout |
| **Final-Decision-9** | Streamlit OOP Architecture | Base/Extension apps, UX standards |
| **Final-Decision-10** | Task-0 Agent Naming | `SnakeAgent` (not `LLMSnakeAgent`) consistency |

## 🏆 High-Priority Coherence Checklist

The following cross-cutting decisions are reflected consistently across **all** Final Decision docs:

1. **Configuration Organization** – Universal constants in `ROOT/config/`; extension-specific configs in `extensions/common/config/` (see FD-2).
2. **Validation Folder** – Comprehensive validators in `extensions/common/validation/` (see FD-2).
3. **Singleton Usage** – Global managers implemented with `SingletonABCMeta` (see FD-3).
4. **Dataset/Model Paths** – `logs/extensions/{datasets|models}/grid-size-N/...` (see FD-1).
5. **Agent Naming** – `agent_*.py` files and `*Agent` classes (see FD-4 & FD-10).
6. **Path Utilities** – Mandatory use of `extensions/common/path_utils.py` (see FD-6).
7. **Grid-Size Agnostic CSV Schema** – 16 normalized features (see FD-1 & FD-2).
8. **Factory & Streamlit Patterns** – Standardized factories (FD-7/8) and OOP Streamlit apps (FD-9).

## ✍️ Editing Policy

* Only modify **Final Decision files** when a new *final* architectural decision has been reached.  Minor clarifications belong in lower-level docs.
* Keep each document focused; link to others rather than duplicate large sections.
* Preserve rich explanations—motivation, trade-offs, design patterns—over excessive code listings.

---

> **In short:** Use this index to navigate the authoritative decisions.  When in doubt, the numbered Final Decision documents outrank every other piece of documentation except a newer Final Decision document with a higher revision number.