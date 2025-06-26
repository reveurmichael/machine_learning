

### **4. Standalone Principle Implementation Gaps**

**Problem**: Multiple interpretations of what constitutes "standalone" behavior.



### **5. Path Management Fragmentation**

**Problem**: Multiple path management approaches mentioned across files.

**Issues**:
- Some files reference manual path construction
- Others mention `unified-path-management-guide.md` 
- Inconsistent `chdir()` usage patterns

**Solution**:
- **Single Authority**: `unified-path-management-guide.md` is the definitive reference
- **Mandatory Pattern**: All extensions MUST use `extensions/common/path_utils.py`
- **Deprecate Manual**: Remove all references to manual path construction

### **6. Factory Pattern Implementation Variations**

**Problem**: Different factory pattern implementations suggested across files.

**Issues**:
- Some files show basic factory patterns
- Others reference complex layered architectures
- Inconsistent agent registration approaches

**Solution**:
- **Unified Reference**: `unified-factory-pattern-guide.md` is authoritative
- **Standard Implementation**: All extensions use the same factory pattern
- **Clear Examples**: Provide consistent factory examples across all documentation


do it for all attached md files that touch this topic









### **7. NPZ Format Specification Gaps**

**Problem**: Unclear NPZ format specifications for different use cases.

**Issues**:
- `data-format-decision-guide.md` mentions NPZ for evolutionary algorithms
- No clear distinction between sequential NPZ and raw NPZ
- Missing specifications for RL experience replay format

**Solution**:
- **Format Specialization**:
  - NPZ Sequential: For RNN/LSTM temporal data
  - NPZ Spatial: For CNN 2D array data  
  - NPZ Raw: For evolutionary algorithms with genetic operators
  - NPZ Experience: For RL experience replay buffers
- **Clear Selection Criteria**: When to use each NPZ variant

do it for all attached md files that touch this topic







### **8. Grid-Size Directory Structure Enforcement**

**Problem**: Inconsistent enforcement of grid-size directory organization.

**Issues**:
- Some files assume fixed grid sizes
- Others mention grid-size agnostic approaches
- Path validation inconsistencies

**Solution**:
- **Universal Grid-Size Paths**: All datasets and models use `grid-size-N/` structure
- **Validation Requirements**: Automatic path validation for compliance
ATETNTION: IN the md files text,  **No NEED TO MENTION Migration Path**: We don't need a process for transitioning existing data.


do it for all attached md files that touch this topic





### **11. Extension Naming Convention Ambiguity**

**Problem**: Inconsistent extension naming patterns across documentation.

**Issues**:
- Some files use `algorithm-v0.0N` format
- Others use `algorithm_v0_0N` format
- Import statement variations

**Solution**:
- **Standard Format**: `algorithm-v0.0N` for directory names
- **Import Format**: Use proper Python import syntax
- **Consistent References**: All documentation uses the same naming convention



do it for all attached md files that touch this topic
