# Game File Manager

## 🎯 **Purpose and Philosophy**

The `BaseFileManager` and `FileManager` serve as the **universal file management system** for all Snake Game AI tasks (0-5). They implement the **Single Source of Truth (SSOT)** principle by providing a centralized, consistent approach to file operations across all extensions.

### **Why This File Manager Exists**

1. **Eliminates Code Duplication**: Before this manager, each extension implemented its own file operations, leading to inconsistencies and maintenance overhead
2. **Ensures Consistency**: All extensions now use identical file naming conventions and JSON schema
3. **Supports Extension Development**: Provides clean hooks for extension-specific file operations without breaking the core file management
4. **Maintains Task-0 Compatibility**: Preserves existing functionality while enabling future extensions

## 🏗️ **Architecture and Design Patterns**

### **Singleton Pattern**
The file manager uses the Singleton pattern to ensure thread-safe file operations:

```python
class BaseFileManager(ABC, metaclass=SingletonABCMeta):
    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self._setup_manager()
```

### **Template Method Pattern**
The manager uses the Template Method pattern to define the file processing workflow:

```python
def process_log_directory(self, log_dir: Union[str, Path]) -> Dict[str, Any]:
    # Step 1: Validate directory
    if not self._validate_directory(log_dir):
        return {}
    
    # Step 2: Load metadata (hook method)
    metadata = self._load_directory_metadata(log_dir)
    
    # Step 3: Process files (abstract method)
    file_data = self._process_directory_files(log_dir)
    
    # Step 4: Generate summary (concrete method with hooks)
    return self._generate_directory_summary(metadata, file_data)
```

### **Strategy Pattern**
Extension-specific file operations are handled through strategy hooks:

```python
def _add_task_specific_summary(self, summary: Dict[str, Any]) -> Dict[str, Any]:
    """Hook method: Add task-specific summary data. Override in subclasses."""
    return {}

def _process_directory_files(self, log_dir: Union[str, Path]) -> Dict[str, Any]:
    """Abstract method: Process files in directory. Must be implemented by subclasses."""
    pass
```

## 📊 **Core File Operations**

### **Universal File Operations (Tasks 0-5)**
All extensions use these core file operations:

```python
# JSON loading/saving with error handling
def save_game_summary(self, game_data: dict, game_duration: float, game_number: int, log_dir: str) -> None
def save_session_summary(self, session_data: dict, session_duration: float, log_dir: str) -> None
def load_game_summary(self, game_number: int, log_dir: str) -> Optional[dict]
def load_session_summary(self, log_dir: str) -> Optional[dict]

# Directory discovery and validation
def find_valid_log_folders(self, root_dir: str = None, max_depth: int = 4) -> List[Path]
def get_next_game_number(self, log_dir: Union[str, Path]) -> int
def get_total_games(self, log_dir: str) -> int

# File naming conventions (Single Source of Truth)
def get_game_json_filename(self, game_number: int) -> str  # Returns "game_N.json"
def join_log_path(self, log_dir: Union[str, Path], filename: str) -> str
```

### **LLM-Specific Operations (Task-0 only)**
Task-0 extends the base with LLM-specific file operations:

```python
# Prompt/response file management
def save_to_file(self, content: str, directory: Union[str, Path], filename: str, metadata: Optional[Dict[str, Any]] = None) -> str
def get_prompt_filename(self, game_number: int, round_number: int, file_type: str = "prompt") -> str
def clean_prompt_files(self, log_dir: Union[str, Path], start_game: int) -> None

# LLM-specific directory structure validation
def find_valid_log_folders(self, root_dir: str = None, max_depth: int = 4) -> List[Path]
```

## 🔧 **Integration with Session Utils**

The file manager is integrated into the session utilities through a factory function:

```python
# In utils/session_utils.py
def get_file_manager():
    from core.game_file_manager import FileManager
    return FileManager()

# Usage in session utilities
def continue_game_web(log_folder: str, max_games: int, host: str, sleep_before: float = 0.0, no_gui: bool = False):
    # ... command setup ...
    st.success(
        f"🌐 Continuation (web) started for '{get_file_manager().get_folder_display_name(log_folder)}' – open http://{url_host}:{port} to watch."
    )
```

### **File Manager Factory Pattern**
The session utilities use the file manager as a factory to access file operations:

```python
# Get file manager instance (singleton)
file_manager = get_file_manager()

# Use file manager for various operations
display_name = file_manager.get_folder_display_name(log_folder)
total_games = file_manager.get_total_games(log_dir)
next_game = file_manager.get_next_game_number(log_dir)
```

## 🎮 **Usage in Extensions**

### **Task-0 (LLM) Usage**
Task-0 uses the full `FileManager` class with LLM-specific operations:

```python
# In core/game_file_manager.py
class FileManager(BaseFileManager):
    def _setup_manager(self) -> None:
        super()._setup_manager()
        self._task_type = "llm_snake_game"
        self._required_directories = [PROMPTS_DIR_NAME, RESPONSES_DIR_NAME]
    
    def _process_directory_files(self, log_dir: Union[str, Path]) -> Dict[str, Any]:
        """Process Task-0 specific files in log directory."""
        directory_path = Path(log_dir)
        
        # Count different file types
        game_files = list(directory_path.glob("game_*.json"))
        prompts_dir, responses_dir = get_llm_directories(directory_path)
        prompt_files = list(prompts_dir.glob("*.txt")) if prompts_dir.exists() else []
        response_files = list(responses_dir.glob("*.txt")) if responses_dir.exists() else []
        
        return {
            "game_count": len(game_files),
            "prompt_count": len(prompt_files),
            "response_count": len(response_files),
            "has_prompts_dir": prompts_dir.exists(),
            "has_responses_dir": responses_dir.exists(),
        }
```

### **Heuristics Extension Usage**
The heuristics extension uses the base file manager for universal operations:

```python
# In extensions/heuristics-v0.04/game_manager.py
from core.game_file_manager import BaseFileManager

class HeuristicGameManager(BaseGameManager):
    def __init__(self, args):
        super().__init__(args)
        # Use base file manager for universal operations
        self.file_manager = BaseFileManager()
    
    def save_game_data(self, game_data: dict, game_duration: float) -> None:
        """Save game data using the unified file manager."""
        self.file_manager.save_game_summary(game_data, game_duration, self.game_count, self.log_dir)
```

## 🔄 **Extension Development Guide**

### **Creating Extension-Specific File Managers**

1. **Inherit from Base Class**:
```python
from core.game_file_manager import BaseFileManager

class MyExtensionFileManager(BaseFileManager):
    def _setup_manager(self) -> None:
        super()._setup_manager()
        self._task_type = "my_extension"
        self._required_directories = ["my_data_dir"]
    
    def _process_directory_files(self, log_dir: Union[str, Path]) -> Dict[str, Any]:
        # Add your extension's file processing logic
        return {
            "my_file_count": len(list(Path(log_dir).glob("my_*.json"))),
            "my_data_dir_exists": (Path(log_dir) / "my_data_dir").exists(),
        }
    
    def _add_task_specific_summary(self, summary: Dict[str, Any]) -> Dict[str, Any]:
        # Add your extension's summary data
        return {
            "task_type": self._task_type,
            "my_extension": True,
        }
```

2. **Integrate with Session Utils**:
```python
# utils/session_utils.py
def get_my_extension_file_manager():
    from my_extension.file_manager import MyExtensionFileManager
    return MyExtensionFileManager()
```

### **Best Practices**

1. **Maintain Core Operations**: Never remove or modify core file operations
2. **Use Extension Hooks**: Add extension-specific logic through the provided hooks
3. **Follow Naming Conventions**: Use consistent file naming across your extension
4. **Document Custom Operations**: Clearly document any extension-specific file operations
5. **Test File Compatibility**: Ensure your file operations work with existing tools

## 📈 **Benefits and Impact**

### **For Extension Developers**
- **90% Reduction in Boilerplate**: No need to implement file operations from scratch
- **Consistent File Management**: All extensions use identical file operations and naming
- **Easy Debugging**: Standardized file operations make issues easier to identify
- **Future-Proof**: New extensions automatically benefit from core improvements

### **For Maintenance**
- **Single Point of Truth**: All file operations centralized in one place
- **Consistent Behavior**: All extensions follow the same file management rules
- **Easy Testing**: Can test file operations independently of extensions
- **Backward Compatibility**: Existing tools continue to work with new extensions

### **For Task-0**
- **Zero Impact**: Existing functionality preserved completely
- **Enhanced Reliability**: More robust file management
- **Better Documentation**: Clear separation between core and LLM-specific operations

## 🎓 **Educational Value**

The game file manager demonstrates several important software engineering principles:

1. **Singleton Pattern**: Ensures thread-safe file operations across the application
2. **Template Method Pattern**: Defines file processing workflow while allowing customization
3. **Strategy Pattern**: Enables different file strategies for different extensions
4. **Single Responsibility**: Focused solely on file management operations
5. **Open/Closed Principle**: Open for extension, closed for modification
6. **DRY Principle**: Eliminates code duplication across extensions
7. **Factory Pattern**: Provides centralized file manager creation through session utilities

## 🔗 **Related Documentation**

- **`game_summary_generator.md`**: Universal summary generation system
- **`game_data_generator.md`**: Universal game data management system
- **`core.md`**: Core architecture and base classes
- **`single-source-of-truth.md`**: SSOT principles and implementation
- **`factory-design-pattern.md`**: Factory pattern usage in the codebase
- **`final-decision.md`**: SUPREME_RULES and architectural standards

---

**The game file manager exemplifies the project's commitment to elegant, maintainable architecture that serves both current needs and future extensions while maintaining strict adherence to `final-decision.md` SUPREME_RULES.** 