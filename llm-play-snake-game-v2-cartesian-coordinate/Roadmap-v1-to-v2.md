# 🚀 Migration Roadmap: From **v1 (Image-Coordinate)** to **v2 (Cartesian-Coordinate)**

> **Folders**  
> • v1 code-base: `./llm-play-snake-game-v1-image-coordinate/`  
> • v2 code-base: `./llm-play-snake-game-v2-cartesian-coordinate/`

---

## 1. Executive Summary

The transition from **v1** to **v2** centers on two fundamental changes: adopting a *Cartesian coordinate system* and enforcing *structured JSON I/O contracts* with the LLM. These changes address core reliability issues discovered in v1 and establish a foundation for more robust human-AI interaction.

**Primary Changes:**
1. 🔄 **Coordinate System**: Switched from image-style `(row, col)` with y-down to mathematical `(x, y)` with y-up
2. 🛡 **Structured I/O**: Replaced regex-based parsing with JSON-first response handling

---

## 2. Motivation: Problems Discovered in v1

| Issue | Manifestation | Impact |
|-------|---------------|--------|
| **Coordinate Confusion** | Image-style `(row, col)` with y-down semantics | • Mental model mismatch for humans and LLMs<br>• Frequent off-by-one errors<br>• Complex coordinate translation in every function |
| **Parsing Brittleness** | Regex patterns like `(\d+)\.?\s+(UP\|DOWN\|LEFT\|RIGHT)` | • Silent failures on malformed LLM output<br>• Game state corruption from unparseable responses<br>• No validation of move sequences |

---

## 3. Technical Changes

### 3.1 Coordinate System Transformation

**v1 Implementation:**
```python
DIRECTIONS = {
    "UP":    (0, -1),  # y decreases (image-style)
    "RIGHT": (1, 0),
    "DOWN":  (0, 1),   # y increases 
    "LEFT":  (-1, 0)
}
```

**v2 Implementation:**
```python
DIRECTIONS = {
    "UP":    (0, 1),   # y increases (Cartesian)
    "RIGHT": (1, 0),
    "DOWN":  (0, -1),  # y decreases
    "LEFT":  (-1, 0)
}
```

**Key Decision:** Origin `(0,0)` moves to bottom-left, aligning with mathematical conventions and reducing cognitive load during debugging.

### 3.2 I/O Contract Evolution

**v1 Approach:** Free-form numbered lists requiring complex regex parsing:
```python
# Multiple fallback patterns in parse_llm_response()
numbered_list = re.findall(r'(\d+)\.?\s+(UP|DOWN|LEFT|RIGHT)', response)
step_pattern = re.findall(r'Step\s+(\d+):\s+(UP|DOWN|LEFT|RIGHT)', response)
```

**v2 Approach:** JSON-first with graceful degradation:
```python
json_data = extract_valid_json(response)
if json_data and validate_json_format(json_data):
    self.planned_moves = json_data["moves"]
```

**Benefits Realized:**
- Single parsing code path for success cases
- Structured validation of response format
- Serializable responses for logging and replay

### 3.3 New Architecture: Defensive Programming

**Runtime Verification System:**
```python
def _verify_coordinate_system(self):
    """Logs coordinate system rules and validates test moves"""
    
def _validate_move(self, current_pos, new_pos, direction_key):
    """Catches coordinate system violations at runtime"""
```

**Multi-Layer Error Recovery:**
1. Direct JSON parsing
2. Preprocessed parsing (handles LLM syntax issues)
3. Array extraction from malformed responses
4. Move sequence validation and filtering

---

## 4. Prompt Engineering Improvements

### 4.1 Template-Based Dynamic Prompts

**v1:** Static prompt with basic string formatting
**v2:** Placeholder-based templates with runtime substitution:

```python
prompt = PROMPT_TEMPLATE_TEXT
prompt = prompt.replace("TEXT_TO_BE_REPLACED_HEAD_POS", head_pos)
prompt = prompt.replace("TEXT_TO_BE_REPLACED_APPLE_POS", apple_pos)
```

### 4.2 Mathematical Coaching

v2 includes explicit move calculations to guide LLM reasoning:
```
"RIGHT moves minus LEFT moves should equal 3 (= 7 - 4)"
"UP moves minus DOWN moves should equal 2 (= 6 - 4)"
```

This addresses discovered limitations in LLM spatial reasoning.


---

## 7. Unexpected Discoveries

### 7.1 LLM Behavioral Patterns
- **Syntax vs. Semantics Gap**: LLMs like deepseek-r1:7b, 14b and 32b can reason about complex game strategies but frequently fail on basic JSON syntax
- **Mathematical Coaching Effectiveness**: Explicit calculations significantly improve spatial reasoning

### 7.2 Architecture Insights
- **Coordinate verification** catches more bugs than anticipated
- **Template-based prompts** enable A/B testing of reasoning strategies
- **JSON responses** accidentally solve future replay requirements

---

## 8. Lessons for AI Practitioners

### 8.1 Technical Lessons
1. **Representation alignment matters** – But syntax/semantics gaps require defensive programming
2. **Structure over eloquence** – Machine-readable contracts outperform natural language flexibility
3. **Incremental migration** – v2 reorganizes without discarding v1 patterns, enabling quick validation
4. **LLM-aware architecture** – Traditional software assumptions don't apply to AI-integrated systems

---

## 9. Concluding Thoughts

The v1→v2 migration reveals that **reliable AI integration** requires more than algorithmic improvements—it demands **architectural discipline**. The coordinate system change was straightforward; the real complexity lay in building robust error recovery, validation, and debugging capabilities around an inherently unreliable AI component.

The technical debt from v1's coordinate confusion and parsing brittleness created a cascade of reliability issues. v2's systematic approach to structured I/O, defensive programming, and comprehensive verification establishes patterns that extend beyond this specific game engine.

**Key Insight**: The migration succeeded not by making the LLM more reliable, but by making its failures more visible and recoverable. This represents a mature approach to human-AI collaboration in software systems.

> "*Good abstractions don't hide complexity—they make failure modes debuggable.*" — Learned from this migration


