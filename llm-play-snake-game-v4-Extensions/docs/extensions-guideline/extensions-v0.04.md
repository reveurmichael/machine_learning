# Extensions v0.04: Advanced Data Generation & LLM Integration

## 🎯 **Core Philosophy: Comprehensive Data Generation**

Extensions v0.04 represent the **advanced stage** of algorithm implementation, focusing on comprehensive data generation capabilities for both supervised learning (CSV) and LLM fine-tuning (JSONL). This version demonstrates sophisticated data pipeline management and cross-extension compatibility.

### **Educational Value**
- **Data Pipeline Management**: Understanding complex data generation workflows
- **Cross-Format Compatibility**: Learning to generate multiple data formats
- **LLM Integration**: Preparing data for language model fine-tuning
- **Advanced Architecture**: Sophisticated extension patterns and design

## 🏗️ **v0.04 Architecture Requirements**

### **Mandatory Directory Structure**
```
extensions/{algorithm}-v0.04/
├── __init__.py
├── app.py                         # Streamlit script launcher (SUPREME_RULE NO.5)
├── game_logic.py                  # Enhanced with dataset generation
├── game_manager.py                # Advanced manager with automatic dataset updates
├── game_data.py                   # Enhanced data handling with explanations
├── dataset_generator.py           # ✨ NEW: Comprehensive dataset generation
├── game_rounds.py                 # ✨ NEW: Round management and data extraction
├── agents/                        # Enhanced algorithm implementations
│   ├── __init__.py
│   ├── agent_bfs.py              # Enhanced with JSONL generation
│   ├── agent_astar.py            # Enhanced with JSONL generation
│   ├── agent_dfs.py              # Enhanced with JSONL generation
│   └── agent_hamiltonian.py      # Enhanced with JSONL generation
├── scripts/                       # Backend execution scripts
│   ├── main.py                   # Enhanced CLI with dataset generation
│   └── run_with_timeout.py       # ✨ NEW: Timeout management
└── README.md                      # Comprehensive documentation
```

### **Factory Pattern: Canonical Method is create()**
All v0.04 extensions must use the canonical method name `create()` for instantiation:

```python
class HeuristicAgentFactory:
    """Advanced factory for v0.04 with comprehensive agent support"""
    
    _registry = {
        "BFS": BFSAgent,
        "ASTAR": AStarAgent,
        "DFS": DFSAgent,
        "HAMILTONIAN": HamiltonianAgent,
    }
    
    @classmethod
    def create(cls, algorithm: str, **kwargs):  # CANONICAL create() method per SUPREME_RULES
        """Create agent using canonical create() method following SUPREME_RULES from final-decision.md"""
        agent_class = cls._registry.get(algorithm.upper())
        if not agent_class:
            available = list(cls._registry.keys())
            raise ValueError(f"Unknown algorithm: {algorithm}. Available: {available}")
        print_info(f"[HeuristicAgentFactory] Creating agent: {algorithm}")  # SUPREME_RULES compliant logging
        return agent_class(**kwargs)
```

## 🚀 **Implementation Examples**

### **Enhanced Dataset Generator**
```python
# dataset_generator.py
class DatasetGenerator:
    """
    Comprehensive dataset generator for v0.04.
    
    Design Pattern: Strategy Pattern
    Purpose: Generate both CSV and JSONL datasets with agent-specific formatting
    Educational Value: Shows advanced data pipeline management
    """
    
    def __init__(self, algorithm: str, output_dir: Path, agent=None):
        self.algorithm = algorithm
        self.output_dir = output_dir
        self.agent = agent
        self._csv_writer = None
        self._jsonl_fh = None
        print_info(f"[DatasetGenerator] Initialized for {algorithm}")  # SUPREME_RULES compliant logging
    
    def _open_csv(self):
        """Open CSV file for writing with UTF-8 encoding (SUPREME_RULE NO.7)"""
        csv_path = self.output_dir / f"{self.algorithm}_dataset.csv"
        fh = csv_path.open("w", newline="", encoding="utf-8")  # UTF-8 encoding
        from extensions.common.config.csv_formats import CSV_ALL_COLUMNS
        writer = csv.DictWriter(fh, fieldnames=CSV_ALL_COLUMNS)
        writer.writeheader()
        self._csv_writer = (writer, fh)
        print_info(f"Opened CSV file: {csv_path}", "DatasetGenerator")
    
    def _open_jsonl(self):
        """Open JSONL file for writing with UTF-8 encoding (SUPREME_RULE NO.7)"""
        jsonl_path = self.output_dir / f"{self.algorithm}_dataset.jsonl"
        self._jsonl_fh = jsonl_path.open("w", encoding="utf-8")  # UTF-8 encoding
        print_info(f"Opened JSONL file: {jsonl_path}", "DatasetGenerator")
    
    def _process_single_game(self, game_data: dict):
        """Process single game for dataset generation"""
        # Extract game components
        moves_history = game_data.get("moves", [])
        explanations = game_data.get("move_explanations", [])
        metrics_list = game_data.get("move_metrics", [])
        
        # Create dataset records using round utilities
        dataset_records = create_dataset_records(
            game_data, moves_history, explanations, metrics_list
        )
        
        # Process each record
        for round_num, move, explanation, metrics, game_state in dataset_records:
            try:
                # Delegate to agent for record creation
                record = self._create_jsonl_record(
                    {
                        "game_state": game_state,
                        "move": move,
                        "explanation": explanation,
                        "metrics": metrics,
                        "game_id": game_data.get("game_number", 1),
                        "round_num": round_num,
                    }
                )
                
                # Write to JSONL
                if self._jsonl_fh:
                    jsonl_line = json.dumps(record) + "\n"
                    self._jsonl_fh.write(jsonl_line)
                    self._jsonl_fh.flush()
                
                # Write to CSV
                if self._csv_writer:
                    csv_record = self._create_csv_record(record, step_number=round_num)
                    self._csv_writer[0].writerow(csv_record)
                    self._csv_writer[1].flush()
                    
            except Exception as e:
                print_warning(f"[DatasetGenerator] Error processing record: {e}")
                continue
```

### **Enhanced Game Manager with Automatic Dataset Updates**
```python
# game_manager.py
class HeuristicGameManager(BaseGameManager):
    """
    Advanced game manager for v0.04 with automatic dataset generation.
    
    Design Pattern: Template Method Pattern
    Purpose: Manages game execution with automatic dataset updates
    Educational Value: Shows sophisticated game management with data generation
    """
    
    def __init__(self, algorithm: str, grid_size: int = 10, max_games: int = 1):
        super().__init__(grid_size=grid_size)
        self.algorithm_name = algorithm
        self.max_games = max_games
        self.dataset_generator = None
        print_info(f"[HeuristicGameManager] Initialized with {algorithm}")  # SUPREME_RULES compliant logging
    
    def initialize(self) -> None:
        """Initialize with automatic dataset update capabilities"""
        super().initialize()
        self._setup_logging()
        self._setup_agent()
        
        # Setup dataset generator for automatic updates
        if self.log_dir:
            self._setup_dataset_generator()
    
    def _setup_dataset_generator(self) -> None:
        """Setup dataset generator for automatic updates"""
        self.dataset_generator = DatasetGenerator(
            self.algorithm_name, Path(self.log_dir), agent=self.agent
        )
        
        # Open CSV and JSONL files for writing
        self.dataset_generator._open_csv()
        self.dataset_generator._open_jsonl()
        
        print_info("[HeuristicGameManager] Dataset generator initialized for automatic updates")
    
    def _update_datasets_incrementally(self, games_data: List[Dict[str, Any]]) -> None:
        """Update datasets incrementally after each game"""
        if not self.dataset_generator:
            print_warning("[HeuristicGameManager] No dataset generator available")
            return
        
        for game_data in games_data:
            game_data["game_number"] = self.game_count
            self.dataset_generator._process_single_game(game_data)
```

### **Enhanced Agent with JSONL Generation**
```python
# agents/agent_bfs.py
class BFSAgent(BaseAgent):
    """
    Enhanced BFS agent with JSONL generation capabilities.
    
    Design Pattern: Strategy Pattern
    Purpose: Implements BFS with comprehensive data generation
    Educational Value: Shows agent-level data generation control
    """
    
    def __init__(self, name: str = "BFS"):
        super().__init__(name)
        print_info(f"[BFSAgent] Initialized BFS agent")  # SUPREME_RULES compliant logging
    
    def generate_jsonl_record(self, game_state: dict, move: str, explanation: dict, 
                            game_id: int = 1, round_num: int = 1) -> Dict[str, Any]:
        """
        Generate complete JSONL record for BFS agent.
        
        SSOT: Single method to generate complete JSONL record for base BFS.
        This centralizes all JSONL generation logic in the agent.
        """
        # Validate move against game state
        valid_moves = calculate_valid_moves_ssot(game_state)
        if move not in valid_moves:
            print_warning(f"[BFSAgent] Move validation failed: {move} not in {valid_moves}")
            return None
        
        # Create comprehensive record
        record = {
            "prompt": self._create_prompt(game_state),
            "completion": self._create_completion(move, explanation),
            "metadata": {
                "algorithm": "BFS",
                "game_id": game_id,
                "round_num": round_num,
                "grid_size": game_state.get("grid_size", 10),
                "snake_length": len(game_state.get("snake_positions", [])),
                "head_position": game_state.get("snake_positions", [[]])[-1] if game_state.get("snake_positions") else None,
                "apple_position": game_state.get("apple_position"),
            },
            "explanation": explanation,
            "metrics": explanation.get("metrics", {}),
        }
        
        return record
    
    def _create_prompt(self, game_state: dict) -> str:
        """Create prompt for BFS agent"""
        head_pos = game_state.get("snake_positions", [[]])[-1] if game_state.get("snake_positions") else None
        apple_pos = game_state.get("apple_position")
        grid_size = game_state.get("grid_size", 10)
        
        return f"Snake head at {head_pos}, apple at {apple_pos} on {grid_size}x{grid_size} grid. What move should the snake make?"
    
    def _create_completion(self, move: str, explanation: dict) -> str:
        """Create completion for BFS agent"""
        reasoning = explanation.get("reasoning", "No reasoning provided")
        return f"Move {move}. {reasoning}"
```

## 📊 **v0.04 Standards**

### **Advanced Data Generation Requirements**
- **Dual Format Support**: Generate both CSV and JSONL datasets
- **Agent-Level Control**: Agents control their own data formatting
- **Automatic Updates**: Incremental dataset updates after each game
- **UTF-8 Encoding**: All file operations use UTF-8 encoding (SUPREME_RULE NO.7)

### **Educational Focus**
- **Data Pipeline Management**: Understanding complex data workflows
- **Cross-Format Compatibility**: Learning multiple data format generation
- **LLM Integration**: Preparing data for language model fine-tuning
- **Advanced Architecture**: Sophisticated extension patterns

### **Quality Standards**
- **Working Data Generation**: Both CSV and JSONL generation must work correctly
- **Agent Integration**: Agents must provide comprehensive data formatting
- **Error Handling**: Robust error handling for data generation
- **Performance**: Efficient data generation without blocking game execution

## 📋 **Implementation Checklist**

### **Required Components**
- [ ] **Dataset Generator**: Comprehensive CSV and JSONL generation
- [ ] **Agent Enhancement**: Agents with JSONL generation capabilities
- [ ] **Automatic Updates**: Incremental dataset updates
- [ ] **UTF-8 Encoding**: All file operations use UTF-8 encoding
- [ ] **Error Handling**: Robust error handling for data generation

### **Code Quality**
- [ ] **Working Data Generation**: Both formats generate correctly
- [ ] **Agent Integration**: Agents provide comprehensive formatting
- [ ] **Performance**: Efficient data generation
- [ ] **Error Handling**: Proper error handling and recovery

### **Educational Value**
- [ ] **Data Pipeline Management**: Clear data workflow understanding
- [ ] **Cross-Format Compatibility**: Understanding multiple data formats
- [ ] **LLM Integration**: Learning LLM fine-tuning data preparation
- [ ] **Advanced Architecture**: Sophisticated extension patterns

## 🎓 **Educational Benefits**

### **Learning Objectives**
- **Data Pipeline Management**: Understanding complex data workflows
- **Cross-Format Compatibility**: Learning multiple data format generation
- **LLM Integration**: Preparing data for language model fine-tuning
- **Advanced Architecture**: Sophisticated extension patterns and design

### **Best Practices**
- **Dual Format Support**: Generate multiple data formats efficiently
- **Agent-Level Control**: Give agents control over their data formatting
- **Automatic Updates**: Incremental dataset updates for efficiency
- **UTF-8 Encoding**: Ensure cross-platform compatibility

## 🔗 **Cross-Extension Integration**

### **Data Format Compatibility**
- **CSV Format**: Compatible with supervised learning extensions
- **JSONL Format**: Compatible with LLM fine-tuning extensions
- **UTF-8 Encoding**: Cross-platform compatibility (SUPREME_RULE NO.7)
- **Standardized Paths**: Consistent dataset storage locations

### **Extension Dependencies**
- **Heuristics v0.04**: Generates standardized datasets for other extensions
- **Supervised Learning**: Consumes CSV datasets from heuristics-v0.04
- **LLM Fine-tuning**: Consumes JSONL datasets from heuristics-v0.04
- **Evaluation**: Consistent comparison framework across all extensions

---

**Extensions v0.04 demonstrate advanced data generation capabilities and sophisticated extension patterns, providing comprehensive datasets for both supervised learning and LLM fine-tuning while maintaining educational value and technical excellence.**

## 🔗 **See Also**

- **`final-decision.md`**: SUPREME_RULES governance system and canonical standards
- **`csv-schema.md`**: CSV schema standards and format specifications
- **`data-format-decision-guide.md`**: Data format selection guidelines
- **`datasets-folder.md`**: Dataset organization and storage standards
