# Snake-GTP Roadmap

The single source of truth for where this repository is heading.


---

## 0  Project philosophy – *first-citizen* vs *second-citizen*

1. **Task-0 = LLM Snake** is, and will always remain, the *first-citizen*.  It
   must compile, run, and deliver the flagship "LLM plays Snake" experience at
   the tip of **master**.  All other research tracks must be able to break,
   iterate, or even be deleted without blocking Task-0.
2. **Task-1 → Task-5** are *second-citizens*.  They live in the same mono-repo
   so they can reuse the game engine and GUIs, but they obey a strict
   dependency direction: they import from Task-0 modules (`core/`, `gui/`),
   never the reverse.
3. The guiding slogan appears throughout this roadmap:  
   **"Task-0 first, everything else second."**


---

## 1  Scope of each task

• **Task-0   LLM Snake** – production-grade game loop driven by large-language
  models.  Includes both PyGame and Flask front-ends, experiment logging, and a
  fully-featured replay system.

• **Task-1   Heuristic Agents** *(second-citizen)* – classical search / rule
  based approaches (BFS, Hamiltonian cycle, longest-safe-path, etc.).  They
  double as data generators for later tracks.

• **Task-2   Supervised Learning** *(second-citizen)* – train neural networks on
  heuristic trajectories; baseline models such as MLP or ResNet on a grid
  encoding.

• **Task-3   Reinforcement Learning** *(second-citizen)* – DQN, PPO and other
  actor-critic variants running on an OpenAI-Gym-compatible wrapper of the
  Snake environment.

• **Task-4   LLM Fine-tuning** *(second-citizen)* – fine-tune an instruction- or
  policy-tuned LLM on the heuristic dataset; evaluate zero-shot vs.
  fine-tuned.

• **Task-5   LLM Distillation** *(second-citizen)* – compress the fine-tuned
  LLM into a smaller student (e.g. LoRA distilled or knowledge-distillation to
  a Transformer-XL).


---

## 2  Repository layout

The paths below are canonical.  Omitted files (`…`) follow the same style.

```
llm-play-snake-game-v3-MoE/
│
├─ core/                 # first-citizen game engine – NO LLM imports
│   ├─ agents.py         # SnakeAgent protocol (single-source truth)
│   ├─ …
│
├─ gui/                  # first-citizen PyGame & shared GUI helpers
│   ├─ base_gui.py
│   ├─ game_gui.py
│   ├─ replay_gui.py
│   └─ …
│
├─ llm/                  # first-citizen prompt assembly, LLM clients, parsers
│   ├─ agent_llm.py      # implements SnakeAgent for Task-0
│   └─ …
│
├─ web/                  # first-citizen Flask site (play, human-play, replay)
│   ├─ templates/
│   └─ static/{css,js,img}
│
├─ utils/                # helpers used by both citizen classes
│   └─ …
│
├─ extensions/           # **all second-citizen tracks live here**
│   │
│   ├─ heuristics/       # Task-1
│   │   ├─ __init__.py
│   │   ├─ algorithms/{base.py,bfs.py,…}
│   │   ├─ app.py            # Streamlit playground
│   │   ├─ config.py         # Task-specific settings (imports from /config)
│   │   ├─ gui_heuristics.py # PyGame subclass for live debug
│   │   └─ web/              # Flask mini-site (Blueprint, templates, static)
│   │
│   ├─ supervised/       # Task-2
│   │   ├─ datasets.py | models/ | train.py
│   │   ├─ app.py            # Streamlit trainer & metrics
│   │   ├─ gui_supervised.py # Optional PyGame dataset visualiser
│   │   ├─ config.py         # Task-specific settings (re-uses /config)
│   │   └─ web/
│   │
│   ├─ reinforcement/    # Task-3
│   │   ├─ env_wrappers.py | train_dqn.py | train_ppo.py
│   │   ├─ app.py            # Streamlit for training curves
│   │   ├─ gui_rl.py         # PyGame overlay with Q-values
│   │   ├─ config.py         # Task-specific settings (re-uses /config)
│   │   └─ web/
│   │
│   ├─ llm_finetune/     # Task-4
│   │   ├─ finetune.py | evaluate.py | collate.py
│   │   ├─ app.py            # Streamlit progress dashboard
│   │   ├─ gui_ft.py         # PyGame comparison viewer
│   │   ├─ config.py         # Task-specific settings (re-uses /config)
│   │   └─ web/
│   │
│   └─ distillation/     # Task-5
│       ├─ distil.py | evaluate.py
│       ├─ app.py            # Streamlit distillation monitor
│       ├─ gui_distill.py    # PyGame side-by-side play
│       ├─ config.py         # Task-specific settings (re-uses /config)
│       └─ web/
│
└─ docs/ config/ logs/ …  # unchanged
```

Key points:
* The **top-level** `web/` belongs exclusively to Task-0.
* Each second-citizen folder contains **three** presentation layers:
  1. `app.py` (Streamlit) – rapid prototyping.
  2. `web/` – Flask Blueprint for integrated dashboards.
  3. `gui_*.py` – PyGame subclass for desktop visualisation.
  These files never touch Task-0's GUI.


---

## 3  Cross-cutting contracts & utilities

* **SnakeAgent** – single interface (in `core/agents.py`) that every policy
  implements.  It returns a direction or `None`.
* **Trajectory** dataclass – canonical `(state, action, reward, next_state,
  done)` used by Task-2, 3, 4, 5.  Lives in `extensions/common_types.py`.
* **Gym wrapper** – `extensions/reinforcement/env_wrappers.py` supplies
  `SnakeEnv`, exposing core logic to Stable-Baselines3 and RLlib.
* **Log convention** – every second-citizen writes to `logs/{task_name}/…`.  
  Task-1 logs feed Task-2 (supervised dataset), Task-3 (RL pretraining), and
  Task-4 (LLM fine-tuning); Task-4 outputs in turn feed Task-5.  The folder
  hierarchy ensures cross-task data sharing without polluting Task-0 logs.


---

## 4  Front-end story

• **Task-0** continues to expose:  
  – PyGame window (`python main.py`)  
  – Flask routes at `/` (play), `/human_play`, `/replay`.

• **Second-citizens** each register a Flask Blueprint under
  `/heuristics`, `/supervised`, `/reinforcement`, `/llm_finetune`, `/distillation`.
  Because Blueprints mount their own template folders, there is zero collision
  with the first-citizen site.  Global assets (Bootstrap, logo) may be linked
  via CDN or copied into each task's static folder for full independence.

• The repository deliberately **does not** add a universal "choose agent"
  dropdown to the first-citizen UI; selecting an agent is handled inside each
  task's own Streamlit or Flask interface.

• PyGame subclasses in second-citizens extend `gui/base_gui.py`, allowing rich
  overlays (e.g. Q-value heat-maps) while sharing the event loop.


---

## 5  Immediate next steps

1. **Finish core/agents.py integration** (done).  
2. Generate empty extension sub-packages with `__init__.py`, `app.py`, `web/`,
   `gui_<task>.py` stubs.  
3. Port existing BFS heuristic into Task-1.  
4. Author design docs for supervised data schema.  
5. Keep Task-0 in CI — every pull request must pass `task0_smoke_test`.


---

## 6  Task-by-task deep dive (deliverables, metrics, inter-task contract)

### 6.1  Task-1  Heuristic Agents  *(second-citizen)*

**Deliverables**  
1. A library of at least five deterministic agents – BFS, A*, Hamiltonian
   cycle, Longest-Safe-Path, Wall-Hugger.  
2. `heuristics/app.py` with interactive sliders (search depth, heuristic
   variant) and real-time board render.  
3. `heuristics/web/` Blueprint exposing `/heuristics/benchmark` route that can
   run batch benchmarks and push results to a CSV download.  
4. `logs/heuristics/…` containing JSONL trajectory dumps and a
   `summary.json` with aggregate scores.

**Success metrics**  
* Shortest-path heuristic should achieve ≥80 % apple efficiency on the default
  15×15 grid.  
* Hamiltonian agent should never crash (100 % survival) on infinite length
  setting.

**Risks & mitigations**  
* Search agents may be too slow for real-time FPS → mitigate with profiling
  hooks and optional Cython compilation.

---

### 6.2  Task-2  Supervised Learning  *(second-citizen)*

**Data source** – exclusively consumes `logs/heuristics/…/*.json` generated by
Task-1.  The converter lives in `supervised/datasets.py` and writes NumPy
arrays to `data/supervised/{version}/…`.

**Models**  
* Baseline MLP (2 hidden layers, ReLU)  
* CNN-based ResNet-like model  
* Optional Vision-Transformer small edition

**Training script** – `train.py` supports both CPU and GPU, prints tensorboard
logs to `logs/supervised/…/tensorboard/`.

**Evaluation** – average apples per game and action accuracy vs ground-truth
heuristic path.

---

### 6.3  Task-3  Reinforcement Learning  *(second-citizen)*

**Environment** – `SnakeEnv` with Gym v0.26 API.  Discrete(4) action space,
observation = 11×11 crop around snake head (configurable).

**Algorithms** – DQN, PPO, optionally SAC for continuous extension.  Uses
Stable-Baselines 3; hyper-params tracked by Hydra configs.

**UI** – `gui_rl.py` overlays current Q-values and policy entropy on the
PyGame board.  Streamlit shows live reward curves fetched from TensorBoard.

---

### 6.4  Task-4  LLM Fine-tuning  *(second-citizen)*

**Data** – concatenates Task-1 logs using `collate.py`, converts to
instruction-tuning JSONL (`{prompt, completion}` pairs).

**Training** – LoRA adapters via PEFT, runs on `transformers==4.*`.

**Metrics** – token-level accuracy, average apples per game when plugged back
into `core` through `LLMPolicy`.

---

### 6.5  Task-5  Distillation  *(second-citizen)*

Compress the Task-4 LoRA-tuned model to a 3-4 B parameter student.
Knowledge-distillation loss =  α * CrossEntropy(student, teacher logits) +
β * KL(student, teacher).

UI (`gui_distill.py`) plays teacher vs student side-by-side for subjective
assessment.

---

## 7  Coding conventions & tooling

* **Python 3.10 or plus** everywhere.  Black for formatting, Ruff for linting, Mypy for type-checking.
* **Import style** – absolute imports inside first-citizen packages; relative
  only within a second-citizen sub-package.

---

## 8  Dependency management

* Core deps pinned in `requirements.txt`.  Second-citizens may append extras
  via `extensions/<task>/requirements.txt` and load them dynamically.
* `pip-tools` used to compile lock files; Task-0 lockfile is authoritative.

---

## 9  Data lineage & sharing rules

1. **Write-once** principle – a task never modifies another task's logs; it
   reads them as input and writes new artefacts in its own folder.
2. **Metadata manifest** – each log directory must include a
   `summary.json` describing schema version, git SHA, and hyper-parameters.
3. **Storage hygiene** – large binary artefacts (model checkpoints >100 MB)
   should be pushed to an external storage bucket and referenced via URL to
   keep the repo lean.


---

## 10  Glossary

* **First-citizen** – code that powers LLM Snake (Task-0); must never break.  
* **Second-citizen** – experimental tracks that import core but can be flaky.  
* **Planned moves** – list of future directions returned by an agent.  
* **EMPTY step** – tick/sentinel where no valid move was produced.  
* **NO_PATH_FOUND** – sentinel indicating heuristic failure.

---

*Document version: latest – expanded for maximum detail.*  
**Task-0 first, everything else second.**