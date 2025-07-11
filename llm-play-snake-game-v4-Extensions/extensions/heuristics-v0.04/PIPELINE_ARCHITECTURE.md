# Simplified JSONL Generation Pipeline Architecture

## Overview

The JSONL generation pipeline follows SSOT (Single Source of Truth) and forward-looking architecture principles with maximum simplicity. All logic is centralized in agent classes with the shortest possible pipeline.

## Key Principles

1. **SSOT**: All JSONL generation logic centralized in agent classes
2. **KISS**: Minimal functions and shortest pipeline
3. **Forward-Looking**: No fallbacks, no backward compatibility
4. **Clean Separation**: Generic data extraction vs. agent-specific wording control

## Architecture

### Current (Ultra-Simplified Pipeline)
```
DatasetGenerator._create_jsonl_record()
├── Extract generic data (game_state, explanation, move, etc.)
└── agent.generate_jsonl_record() # ONE METHOD CALL
    ├── All SSOT validation and fail-fast checks
    ├── All metric calculation and feature building
    ├── Prompt/completion wording control
    └── Complete record generation
```

**Total Pipeline**: 2 steps (data extraction + agent delegation)

## Responsibility Separation

### DatasetGenerator (Generic)
- Extract `game_state`, `explanation`, `move_chosen`, `game_id`, `round_num`
- Delegate to agent's centralized method
- **No** validation, formatting, or wording logic

### Agent (Specific) 
- All SSOT validation and fail-fast checks
- All metric calculation and feature building
- Complete control over prompt/completion wording
- All JSONL record generation logic

## Agent Implementation (BFS512TokenAgent)

### Required Method
```python
def generate_jsonl_record(self, game_state, move, explanation, game_id, round_num):
    """SSOT: Complete JSONL record generation with validation."""
    # All validation, formatting, and generation logic here
    return {"prompt": prompt, "completion": completion}
```

### Control Switches (Agent-Specific Wording)
```python
self.include_board_representation = False  # ASCII board in prompt
self.include_danger_assessment = False     # Danger analysis in completion
self.include_apple_direction = False       # Apple direction in completion
self.include_free_space = False           # Free space metrics in completion
self.include_metrics_in_completion = False # Show metrics in completion text
```

## DatasetGenerator Implementation

```python
def _create_jsonl_record(self, record: dict) -> Dict[str, Any]:
    """Create JSONL record using agent's centralized generation method."""
    # Generic data extraction (centralized in dataset_generator)
    game_state = record.get("game_state", {})
    explanation = record.get("explanation", {})
    move_chosen = record.get("move")
    game_id = record.get("game_id", 1)
    round_num = record.get("round_num", 1)

    # Delegate all logic to agent
    return self.agent.generate_jsonl_record(
        game_state, move_chosen, explanation, game_id, round_num
    )
```

## Benefits

1. **Ultra-Short Pipeline**: 2-step process (extract + delegate)
2. **SSOT Compliance**: Single point of validation and generation
3. **Clean Separation**: Generic extraction vs. specific wording control
4. **Forward-Looking**: No legacy code or fallbacks
5. **KISS Principle**: Maximum simplicity with full functionality
6. **Agent Control**: Complete control over prompt/completion formatting

## Extension to Other Agents

Any agent implementing `generate_jsonl_record()` method automatically works with the pipeline:

1. Agent receives extracted generic data
2. Agent handles all validation and formatting
3. Agent controls all wording and features
4. Pipeline remains identical across all agents

This architecture achieves the shortest possible pipeline while maintaining complete functionality and SSOT compliance. 