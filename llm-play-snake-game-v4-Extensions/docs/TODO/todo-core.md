# Task-0 Core Refactoring for Extension-Friendliness

## 🎯 **Core Problem Diagnosis**

While the current architecture is strong, further abstraction and standardization will make it much easier to add new extensions (e.g., heuristics-v0.04) and ensure all outputs (game_N.json, summary.json) are consistent and robust.

---

## 🚩 **Key Problems**

1. **JSON/game summary generation is duplicated**: Each extension implements its own `generate_game_summary()` logic, leading to code repetition and inconsistencies.
2. **Statistics collection is manual and error-prone**: Step stats, time stats, and other metrics are managed by hand in each extension.
3. **File/session management is scattered**: Logic for saving game/session files is spread across multiple classes and not standardized.
4. **Extension logic requires method overrides**: Extensions must override or duplicate methods to add custom logic at various lifecycle points.
5. **Extension-specific data/config validation is inconsistent**: Each extension manages its own data/config validation, leading to inconsistencies.

---

## 🚀 **Unified Refactoring Solution**

### 1. **Universal JSON/Game Summary Generator**
- Implement `BaseGameSummaryGenerator` (e.g., `core/game_summary_generator.py`) using Template Method/Strategy Patterns.
- All tasks/extensions use this generator for summary creation, with extension-specific fields handled via subclassing or hooks.
- Ensures consistency, reduces boilerplate, and supports Task-0 compatibility.

### 2. **Standardized Statistics/Data Collection**
- Implement `GameStatisticsCollector` (e.g., `core/game_statistics_collector.py`) using Observer/Facade Patterns.
- Automatically aggregates and manages all relevant statistics, with extension-specific collectors pluggable as needed.
- Integrate with `BaseGameData` so all moves/game ends are recorded through the collector.

### 3. **Unified File and Session Management**
- Create `UniversalFileManager` (e.g., `core/game_file_manager.py`) to provide a single API for saving game files and session summaries.
- File manager uses the summary generator and stats collector to ensure all outputs are consistent and up-to-date.

### 4. **Extensibility Hooks and Callbacks**
- Add a callback/hook system to `BaseGameManager` (e.g., `register_extension_callback(event, callback)` and `_trigger_extension_callbacks(event, **kwargs)`).
- Extensions register logic for events like `pre_game`, `post_game`, `pre_move`, `post_move`, and `dataset_update`.
- The base manager triggers these hooks at the appropriate times.

### 5. **Extension-Specific Data Storage and Validation**
- Add `extension_data` and related methods to `BaseGameData` for standardized extension-specific storage.
- Implement `ExtensionConfig` and `ExtensionConfigManager` for standardized configuration validation.

---

## 📋 **Implementation Roadmap**

### **Phase 1 (High Priority)**
1. Implement `BaseGameSummaryGenerator` and refactor all summary generation to use it.
2. Integrate `GameStatisticsCollector` into `BaseGameData` and all extensions.
3. Refactor file/session management to use `UniversalFileManager`.
4. Add extensibility hooks/callbacks to `BaseGameManager`.

### **Phase 2 (Medium Priority)**
5. Add extension-specific data storage and validation to `BaseGameData` and `core/extension_config.py`.
6. Standardize data export interfaces for all extensions.

### **Phase 3 (Long-Term)**
7. Add advanced monitoring, dependency management, and performance tracking as needed.

---

## 📈 **Expected Benefits**

- **For Extension Developers:**
  - 50–90% reduction in boilerplate code.
  - Standardized, reliable patterns for all extensions.
  - Faster, easier development and debugging.
- **For Maintenance:**
  - Centralized logic for common operations.
  - Consistent behavior and easier testing.
  - Better documentation and onboarding.
- **For Task-0:**
  - Zero impact on existing functionality.
  - Cleaner, more maintainable, and future-proof codebase.

---

## 🏆 **Heuristics v0.04 Success Highlights**

- Clean extension structure and round management.
- Rich, consistent datasets with detailed explanations.
- No base class pollution or backward compatibility baggage.
- Forward-looking, extensible architecture.

---

**Next Steps:**
1. Review and approve this plan.
2. Implement Phase 1 features in Task-0.
3. Update at least one extension to use the new abstractions.
4. Gradually migrate all extensions.
5. Monitor, iterate, and document best practices.