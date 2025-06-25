# Extension Guidelines Contradictions and Issues Analysis


### 3. **Data Format Decision Matrix Confusion**

**Problem:**
- `csv-schema-1.md` promotes 16-feature CSV as universal format
- `evolutionary.md` says CSV is insufficient for evolutionary algorithms
- `extensions-v0.04.md` suggests both CSV AND JSONL for same heuristics extension
- No clear decision tree for format selection

**Solution:**
```
✅ UNIFIED FORMAT DECISION MATRIX:
Algorithm Type    | Primary Format | Use Case
Heuristics v0.03  | CSV           | Supervised learning training
Heuristics v0.04  | JSONL         | LLM fine-tuning only  
Supervised        | CSV           | Training tabular models
Reinforcement     | NPZ           | Experience replay buffers
Evolutionary      | Raw Arrays    | Population-based optimization

- Remove conflicting statements about universal CSV applicability
- Create single authoritative format selection guide
```

### 4. **Extension Evolution Rules Ambiguity**

**Problem:**
- Multiple files use terms like "copy exactly", "enhancements allowed", "forbidden changes"
- No clear definition of what constitutes a "breaking change"
- Inconsistent rules about when modifications are allowed vs. forbidden

**Solution:**
```
✅ EXPLICIT EVOLUTION RULES:
Breaking Changes (FORBIDDEN v0.02→v0.03→v0.04):
- Modifying core algorithm logic
- Changing factory registration names  
- Removing existing agent files
- Changing public method signatures

Allowed Changes (PERMITTED):
- Adding new agent variants (agent_bfs_enhanced.py)
- Adding monitoring/web utilities
- Adding new methods (not modifying existing)
- Adding performance optimizations that maintain interface
```

### 5. **Path Management Implementation Scatter**

**Problem:**
- Multiple files reference `final-decision-6.md` which isn't provided
- Different files show different path management patterns
- `app.md` shows one pattern, `cwd-and-logs.md` shows another
- Implementation details scattered across multiple documents

**Solution:**
```
✅ CONSOLIDATE PATH MANAGEMENT:
- Create single authoritative path_utils.py implementation guide
- Standardize on one pattern across all extension documents
- Remove scattered implementation examples
- Provide complete working code samples in one location
```


### 7. **Documentation Reference Chain Breaks**

**Problem:**
- Frequent references to "Final Decision Series" (final-decision-0 through final-decision-10)
- Most of these documents are not provided or empty
- Creates broken reference chains throughout guidelines
- "Authoritative Reference" claims that can't be verified

**Solution:**
```
✅ FIX REFERENCE INTEGRITY:
- Remove references to unavailable final-decision documents
- Make each guideline self-contained where possible
- Create actual final-decision documents or remove references
- Establish clear document hierarchy without broken links
```

### 8. **Agent Factory Pattern Duplication**

**Problem:**
- Factory pattern implementation shown differently in multiple files
- `agents.md`, `extensions-v0.02.md`, and others show variations
- No consistent factory interface definition
- Code duplication across documentation

**Solution:**
```
✅ UNIFY FACTORY PATTERN:
- Create single canonical factory interface in extensions/common/
- Show consistent implementation pattern across all algorithm types
- Remove duplicate factory examples from individual guideline files
- Reference single authoritative factory implementation
```

