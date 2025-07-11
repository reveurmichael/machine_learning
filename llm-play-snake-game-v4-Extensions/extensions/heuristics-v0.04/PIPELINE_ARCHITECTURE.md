# Simplified JSONL Generation Pipeline Architecture

## Overview

The JSONL generation pipeline has been redesigned to follow SSOT (Single Source of Truth) principles with maximum simplicity and maintainability. The new architecture centralizes all logic in the agent classes while keeping the pipeline as short as possible.

## Key Principles

1. **SSOT**: All JSONL generation logic centralized in agent classes
2. **KISS**: Minimal functions and short pipeline
3. **Flexibility**: Easy on/off switches for customization
4. **Maintainability**: No code duplication across components

## Architecture

### Before (Complex Multi-Layer Pipeline)
```
DatasetGenerator._create_jsonl_record()
├── Extract positions and features
├── Calculate metrics
├── Validate data
├── Build additional features
├── Format prompt via agent.format_prompt()
├── Format completion via agent.format_completion()
└── Return record
```

### After (Simplified Centralized Pipeline)
```
DatasetGenerator._create_jsonl_record()
└── agent.generate_jsonl_record() # ONE METHOD CALL
    ├── All validation and extraction
    ├── All metric calculation  
    ├── All feature building
    └── Complete record generation
```

## Agent Implementation (BFS512TokenAgent)

### Core Method
```python
def generate_jsonl_record(self, game_state, move, explanation, game_id, round_num):
    """SSOT: Single method to generate complete JSONL record."""
    # All logic centralized here
    return {"prompt": prompt, "completion": completion}
```

### Control Switches
```python
# Prompt customization
self.include_board_representation = False  # ASCII board in prompt

# Completion customization  
self.include_danger_assessment = False     # Danger analysis
self.include_apple_direction = False       # Apple direction info
self.include_free_space = False           # Free space metrics
self.include_metrics_in_completion = False # Metrics in completion text
```

## Usage Examples

### Basic JSONL Generation
```python
agent = BFS512TokenAgent()
# All switches default to False for minimal output

record = agent.generate_jsonl_record(
    game_state=game_state,
    move="UP",
    explanation=explanation,
    game_id=1,
    round_num=1
)
```

### Enhanced JSONL with All Features
```python
agent = BFS512TokenAgent()
agent.include_apple_direction = True
agent.include_danger_assessment = True
agent.include_free_space = True
agent.include_metrics_in_completion = True

record = agent.generate_jsonl_record(...)  # Same call, richer output
```

## Benefits

1. **Shorter Pipeline**: From ~200 lines in DatasetGenerator to 1 method call
2. **Centralized Logic**: All JSONL logic lives in agent classes
3. **Easy Customization**: Simple boolean switches for features
4. **SSOT Compliance**: Single validation and extraction point
5. **Maintainable**: No code duplication between components
6. **Backwards Compatible**: Fallback for older agents

## Dataset Generator Changes

The `DatasetGenerator._create_jsonl_record()` method is now just:

```python
def _create_jsonl_record(self, record):
    if hasattr(self.agent, 'generate_jsonl_record'):
        return self.agent.generate_jsonl_record(...)  # Centralized
    else:
        return self._create_jsonl_record_fallback(...)  # Backwards compatibility
```

## Extension to Other Agents

To add centralized JSONL generation to any agent:

1. Add control switches as class attributes
2. Implement `generate_jsonl_record()` method
3. Move agent-specific logic from DatasetGenerator to agent
4. DatasetGenerator automatically uses the centralized method

This architecture makes the codebase more maintainable while preserving all functionality and flexibility. 