# Final Decision 6: Path Management Standardization

> **SUPREME AUTHORITY**: This document establishes the definitive path management standards for all Snake Game AI extensions.

> **See also:** `unified-path-management-guide.md` (Path standards), `cwd-and-logs.md` (Working directory), `standalone.md` (Extension independence), `final-decision-10.md` (SUPREME_RULES).

## 🎯 **Executive Summary**

This document establishes **mandatory path management standards** for all Snake Game AI extensions, requiring consistent use of `extensions/common/utils/path_utils.py` utilities. This ensures reliable cross-platform operation, consistent working directory management, and eliminates path-related bugs across all extension types, strictly following `final-decision-10.md` SUPREME_RULES.

### **SUPREME_RULES Integration**
- **SUPREME_RULE NO.1**: Enforces reading all GOOD_RULES before making path management changes to ensure comprehensive understanding
- **SUPREME_RULE NO.2**: Uses precise `final-decision-N.md` format consistently when referencing architectural decisions
- **SUPREME_RULE NO.3**: Enables lightweight common utilities with OOP extensibility while maintaining path patterns through inheritance rather than tight coupling
- **SUPREME_RULE NO.4**: Ensures all markdown files are coherent and aligned through nuclear diffusion infusion process

### **GOOD_RULES Integration**
This document integrates with the **GOOD_RULES** governance system established in `final-decision-10.md`:
- **`unified-path-management-guide.md`**: Authoritative reference for path management standards
- **`cwd-and-logs.md`**: Authoritative reference for working directory and log organization
- **`single-source-of-truth.md`**: Ensures path consistency across all extensions
- **`standalone.md`**: Maintains extension independence through proper path management



### **2. Dataset and Model Path Management**

```python
def get_dataset_path(extension_type: str, version: str, grid_size: int, 
                    algorithm: str, timestamp: str = None) -> Path:
    """
    Get standardized dataset path following final-decision-1.md structure
    
    Args:
        extension_type: Type of extension (heuristics, supervised, etc.)
        version: Extension version (0.01, 0.02, 0.03, etc.)
        grid_size: Grid size used for training/generation
        algorithm: Algorithm name (bfs, astar, mlp, etc.)
        timestamp: Session timestamp (auto-generated if None)
        
    Returns:
        Path: Dataset directory path
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    session_name = f"{extension_type}_v{version}_{timestamp}"
    path = Path("logs/extensions/datasets") / f"grid-size-{grid_size}" / session_name / algorithm
    print_info(f"[PathUtils] Generated dataset path: {path}")  # Simple logging
    return path

def get_model_path(extension_type: str, version: str, grid_size: int,
                  model_name: str, timestamp: str = None) -> Path:
    """
    Get standardized model path following final-decision-1.md structure
    
    Args:
        extension_type: Type of extension (supervised, reinforcement, etc.)
        version: Extension version
        grid_size: Grid size used for training
        model_name: Model name (mlp, dqn, lora, etc.)
        timestamp: Session timestamp (auto-generated if None)
        
    Returns:
        Path: Model directory path
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    session_name = f"{extension_type}_v{version}_{timestamp}"
    path = Path("logs/extensions/models") / f"grid-size-{grid_size}" / session_name / model_name
    print_info(f"[PathUtils] Generated model path: {path}")  # Simple logging
    return path

def validate_path_structure(project_root: Path, extension_path: Path) -> None:
    """
    Validate that path structure follows required patterns
    
    Args:
        project_root: Project root directory
        extension_path: Extension directory
        
    Raises:
        ValueError: If path structure is invalid
    """
    print_info(f"[PathUtils] Validating path structure")  # Simple logging
    
    # Validate project root
    if not (project_root / "README.md").exists():
        raise ValueError(f"Invalid project root: {project_root}")
    
    if not (project_root / "core").exists():
        raise ValueError(f"Missing core/ directory in project root: {project_root}")
    
    # Validate extension path
    if not extension_path.exists():
        raise ValueError(f"Extension path does not exist: {extension_path}")
    
    # Validate extension structure
    required_files = ["__init__.py", "game_logic.py", "game_manager.py"]
    for file in required_files:
        if not (extension_path / file).exists():
            raise ValueError(f"Missing required file {file} in extension: {extension_path}")
    
    print_success(f"[PathUtils] Path structure validation passed")  # Simple logging
```

## 🎓 **Educational Value and Learning Path**

### **Learning Objectives**
- **Path Management**: Understanding cross-platform path handling
- **Working Directory**: Learning to manage working directories properly
- **Factory Patterns**: Understanding canonical factory pattern implementation
- **Extension Independence**: Learning to create standalone extensions

### **Implementation Examples**
- **Extension Setup**: How to set up extensions with proper path management
- **Path Generation**: How to generate consistent paths across extensions
- **Validation**: How to validate path structures
- **Factory Integration**: How to integrate path management with factory patterns

## 🔗 **Integration with Other Documentation**

### **GOOD_RULES Alignment**
This document aligns with:
- **`unified-path-management-guide.md`**: Detailed path management standards
- **`cwd-and-logs.md`**: Working directory and log organization
- **`standalone.md`**: Extension independence principles
- **`single-source-of-truth.md`**: Architectural principles

### **Extension Guidelines**
This path management system supports:
- All extension types (heuristics, supervised, reinforcement, LLM)
- All extension versions (v0.01, v0.02, v0.03)
- Cross-platform compatibility
- Consistent path organization across all extensions

---

**This path management standardization ensures reliable, consistent, and cross-platform path handling across all Snake Game AI extensions while maintaining SUPREME_RULES compliance.**

> **SUPREME_RULES COMPLIANCE**: This document strictly follows the SUPREME_RULES established in `final-decision-10.md`, ensuring coherence, educational value, and architectural integrity across the entire project. 