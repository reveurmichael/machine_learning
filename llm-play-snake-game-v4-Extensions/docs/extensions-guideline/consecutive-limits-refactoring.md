# Consecutive Limits Refactoring for Snake Game AI

> **Important — Authoritative Reference:** This document supplements the _Final Decision Series_ (`final-decision-0.md` → `final-decision-10.md`) and defines consecutive limits refactoring patterns.

> **See also:** `core.md`, `final-decision-10.md`, `project-structure-plan.md`.

# Elegant Consecutive Limits Management System

## Overview

This document describes the elegant refactoring of consecutive move limits and sleep handling in the Snake-GTP project. The system centralizes all limit tracking into a sophisticated management architecture that follows OOP, SOLID, and DRY principles while strictly adhering to `final-decision-10.md` SUPREME_RULES.

## Architecture

### Core Components

#### 1. `core/game_limits_manager.py` (Main System)
**Design Patterns Used:**
- **Facade Pattern**: `ConsecutiveLimitsManager` provides a simple interface to complex limit tracking
- **Strategy Pattern**: `LimitEnforcementStrategy` allows pluggable enforcement policies
- **Template Method Pattern**: Consistent move processing workflow
- **Value Object Pattern**: Immutable `LimitStatus` and `LimitConfiguration` classes

**Key Classes:**
- `ConsecutiveLimitsManager`: Central management system for all consecutive and absolute limits
- `LimitType`: Enum defining the five limit types (EMPTY, SOMETHING_IS_WRONG, INVALID_REVERSALS, NO_PATH_FOUND, MAX_STEPS)
- `LimitConfiguration`: Immutable configuration with validation
- `LimitStatus`: Thread-safe status tracking
- `StandardLimitEnforcement`: Default enforcement strategy

#### 2. `core/game_state_adapter.py` (Interface Adapter)
**Design Patterns Used:**
- **Adapter Pattern**: Bridges interface gaps between different game state representations
- **Factory Pattern**: `create_game_state_adapter()` for consistent creation

**Purpose:**
- Eliminates code duplication across multiple files
- Provides consistent interface for game state access
- Centralizes game state logic

### Limit Types Managed

1. **EMPTY Moves** (`--max-consecutive-empty-moves-allowed`)
   - Tracks when LLM returns no valid move
   - Configurable sleep penalty after empty moves
   - Intelligent reset on valid moves

2. **SOMETHING_IS_WRONG** (`--max-consecutive-something-is-wrong-allowed`)
   - Handles LLM parsing/communication errors
   - Sophisticated error tracking and recovery
   - Progressive warnings before limits

3. **INVALID_REVERSALS** (`--max-consecutive-invalid-reversals-allowed`)
   - Prevents snake from moving backwards
   - Smart detection of reversal attempts
   - Graceful handling with user feedback

4. **NO_PATH_FOUND** (`--max-consecutive-no-path-found-allowed`)
   - Tracks pathfinding failures
   - Handles maze-like situations
   - Intelligent counter management

5. **MAX_STEPS** (`--max-steps`)
   - Prevents infinite games by limiting total steps
   - Progressive warnings at 90% of limit
   - Elegant termination with proper game end recording

## Integration Points

### GameManager Integration
```python
# In core/game_manager.py
from core.game_limits_manager import create_limits_manager
self.limits_manager = create_limits_manager(args)
```

### Game Loop Integration
```python
# In core/game_loop.py - handles EMPTY, NO_PATH_FOUND, INVALID_REVERSALS, MAX_STEPS
# Uses shared GameStateAdapter for consistent interface
# Replaces scattered BaseGameManagerHelper.check_max_steps() calls
```

### Communication Utils Integration
```python
# In llm/communication_utils.py - handles SOMETHING_IS_WRONG
# Elegant exception handling with sophisticated tracking
```

## Key Features

### 1. Intelligent Counter Management
- **Progressive Warnings**: Alerts at 75% of limit thresholds
- **Smart Resets**: Counters reset based on move success patterns
- **Context-Aware**: Different reset logic for different limit types

### 2. Sophisticated Sleep Management
- **Configurable Penalties**: Sleep duration based on limit type
- **Elegant Logging**: Informative messages during sleep periods
- **Performance Optimization**: Only sleeps when necessary
- **No Sleep for Absolute Limits**: MAX_STEPS doesn't trigger sleep penalties

### 3. Comprehensive Status Tracking
- **Real-time Monitoring**: Current status of all limit types
- **Historical Data**: Track patterns over time
- **Detailed Reporting**: Rich status information for debugging

### 4. Educational Design Patterns
- **Multiple Patterns**: Demonstrates 6+ design patterns in action
- **Clean Architecture**: SOLID principles throughout
- **Extensible Design**: Easy to add new limit types or strategies

## Configuration

All limits are configurable via command-line arguments:

```bash
python scripts/main.py \
  --max-consecutive-empty-moves-allowed 3 \
  --max-consecutive-something-is-wrong-allowed 2 \
  --max-consecutive-invalid-reversals-allowed 5 \
  --max-consecutive-no-path-found-allowed 1 \
  --max-steps 1000 \
  --sleep-after-empty-step 0.5
```

## Benefits Achieved

### Code Quality
- ✅ **DRY Principle**: Eliminated duplicate limit checking across 8+ files
- ✅ **SOLID Principles**: Clean separation of concerns and responsibilities
- ✅ **OOP Design**: Proper encapsulation and inheritance hierarchies
- ✅ **Design Patterns**: Educational implementation of multiple patterns

### User Experience
- ✅ **Progressive Warnings**: Users get advance notice before limits are reached
- ✅ **Informative Messages**: Clear, actionable error messages
- ✅ **Configurable Behavior**: Fine-tuned control over limit enforcement
- ✅ **Elegant Sleep Management**: Sophisticated timing with user feedback

### System Architecture
- ✅ **Centralized Management**: Single source of truth for all limits
- ✅ **Thread-Safe Operations**: Concurrent access handled properly
- ✅ **Extensible Design**: Easy to add new limit types or enforcement strategies
- ✅ **Clean Integration**: Seamless integration with existing codebase

## Implementation Notes

### Following System-Prompt.txt Guidelines
- **File Naming**: Uses `game_limits_manager.py` to follow `game_*.py` pattern in core folder
- **No Backward Compatibility**: Clean forward-looking approach, no deprecated code
- **Direct Architecture**: Uses `FileManager` directly instead of utility wrappers
- **Documentation First**: Comprehensive comments and docstrings throughout

### Design Philosophy
This refactoring transforms scattered, error-prone limit checking into a maintainable, extensible architecture while serving as an educational example of how design patterns solve real-world problems effectively.

The system demonstrates that elegant code is not just about aesthetics—it's about creating robust, maintainable solutions that are both sophisticated and approachable for future developers.

**Unified Limit Management**: The system elegantly handles both consecutive limits (EMPTY, SOMETHING_IS_WRONG, etc.) and absolute limits (MAX_STEPS) using the same architectural patterns, providing consistency while accommodating different limit semantics.

## Future Extensions

The architecture is designed to support future enhancements:
- Additional limit types can be easily added to the `LimitType` enum
- New enforcement strategies can implement the `LimitEnforcementStrategy` interface
- Custom sleep behaviors can be plugged in via the strategy pattern
- Advanced analytics and reporting can be added to the status tracking system

## 🔗 **See Also**

- **`core.md`**: Base class architecture and inheritance patterns
- **`final-decision-10.md`**: SUPREME_RULES governance system and canonical standards
- **`project-structure-plan.md`**: Project structure and organization 