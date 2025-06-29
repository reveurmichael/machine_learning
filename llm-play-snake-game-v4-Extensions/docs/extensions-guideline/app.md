# Streamlit Application Architecture for Snake Game AI Extensions

> **Important — Authoritative Reference:** This document supplements the _Final Decision Series_ (`final-decision-0.md` → `final-decision-10.md`) and defines Streamlit application architecture patterns.

> **See also:** `scripts.md`, `dashboard.md`, `final-decision-10.md`, `standalone.md`.

## 🎯 **Core Philosophy: Interactive Algorithm Exploration**

Streamlit applications in Snake Game AI extensions serve as **interactive dashboards** that enable researchers, students, and developers to explore algorithm behavior, compare performance, and understand the decision-making processes of different AI approaches.

### **Educational Value**
- **Algorithm Visualization**: Real-time observation of decision-making processes
- **Interactive Parameter Tuning**: Dynamic exploration of algorithm behavior
- **Comparative Analysis**: Side-by-side algorithm performance evaluation
- **Learning Analytics**: Progress tracking and performance insights

## 🏗️ **Streamlit Architecture Patterns**

### **Universal v0.03 Requirements**

All extensions v0.03 must implement Streamlit applications following these standards:

#### **Application Structure**
```
extensions/{algorithm}-v0.03/
├── app.py                          # 🎯 Main Streamlit application entry point
├── dashboard/                      # 📁 Dashboard component modules
│   ├── __init__.py                 # Dashboard package initialization
│   ├── components/                 # 📁 Reusable UI components
│   │   ├── __init__.py
│   │   ├── algorithm_selector.py   # Algorithm selection interface
│   │   ├── parameter_controls.py   # Parameter adjustment widgets
│   │   ├── progress_monitor.py     # Training/execution progress display
│   │   └── results_display.py      # Results visualization components
│   ├── pages/                      # 📁 Multi-page application structure
│   │   ├── __init__.py
│   │   ├── training.py             # Training interface and monitoring
│   │   ├── evaluation.py           # Model evaluation and testing
│   │   ├── comparison.py           # Algorithm comparison dashboard
│   │   └── analysis.py             # Performance analysis and insights
│   └── utils/                      # 📁 Dashboard-specific utilities
│       ├── __init__.py
│       ├── session_state.py        # Session state management
│       ├── config_loader.py        # Configuration file handling
│       └── subprocess_runner.py    # Safe subprocess execution
└── scripts/                        # 📁 Backend execution scripts
    ├── main.py                     # Core algorithm execution
    ├── train.py                    # Training pipeline (if applicable)
    └── evaluate.py                 # Evaluation utilities
```

### **Path Management Integration**

Streamlit applications must integrate with the established path management system:

```python
# extensions/{algorithm}-v0.03/app.py

# MANDATORY: Ensure proper working directory before any imports
import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

# Now safe to import project modules
import streamlit as st
from extensions.common.utils.path_utils import get_extension_path, ensure_logs_directory
from dashboard.components import AlgorithmSelector, ParameterControls, ProgressMonitor
```

### **Base Application Template**

All Streamlit applications should inherit from a common base structure:

```python
class BaseExtensionApp:
    """
    Base class for Streamlit extension applications
    
    Design Pattern: Template Method Pattern
    Purpose: Provide consistent structure and functionality across all extensions
    Educational Value: Demonstrates how template patterns enable code reuse
    while allowing customization of specific behaviors.
    """
    
    def __init__(self, extension_name: str):
        self.extension_name = extension_name
        self.setup_page_config()
        self.initialize_session_state()
        self.setup_sidebar()
        print(f"[{extension_name}App] Initialized")  # Simple logging
        
    def setup_page_config(self):
        """Configure Streamlit page settings"""
        st.set_page_config(
            page_title=f"{self.extension_name} Dashboard",
            page_icon="🐍",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'algorithm' not in st.session_state:
            st.session_state.algorithm = self.get_default_algorithm()
        if 'grid_size' not in st.session_state:
            st.session_state.grid_size = 10
        if 'max_games' not in st.session_state:
            st.session_state.max_games = 1
        if 'running' not in st.session_state:
            st.session_state.running = False
            
    def setup_sidebar(self):
        """Setup common sidebar elements"""
        with st.sidebar:
            st.title(f"{self.extension_name} Control Panel")
            
            # Algorithm selection
            algorithms = self.get_available_algorithms()
            st.session_state.algorithm = st.selectbox(
                "Algorithm",
                algorithms,
                index=algorithms.index(st.session_state.algorithm)
            )
            
            # Common parameters
            st.session_state.grid_size = st.slider("Grid Size", 5, 25, st.session_state.grid_size)
            st.session_state.max_games = st.number_input("Max Games", 1, 100, st.session_state.max_games)
            
            # Extension-specific controls
            self.add_extension_controls()
    
    def add_extension_controls(self):
        """Add extension-specific sidebar controls (implemented by subclasses)"""
        pass
        
    def get_available_algorithms(self) -> List[str]:
        """Return list of available algorithms (implemented by subclasses)"""
        raise NotImplementedError
        
    def get_default_algorithm(self) -> str:
        """Return default algorithm name (implemented by subclasses)"""
        return self.get_available_algorithms()[0]
```

## 🔧 **Extension-Specific Implementations**

### **Heuristics Dashboard**

```python
class HeuristicsApp(BaseExtensionApp):
    """
    Streamlit dashboard for heuristic pathfinding algorithms
    
    Features:
    - Algorithm selection (BFS, A*, DFS, Hamiltonian)
    - Real-time pathfinding visualization
    - Performance comparison tools
    - Path optimality analysis
    """
    
    def __init__(self):
        super().__init__("Heuristics")
        self.main()
    
    def get_available_algorithms(self) -> List[str]:
        return ["BFS", "ASTAR", "DFS", "HAMILTONIAN"]
    
    def add_extension_controls(self):
        """Add heuristics-specific controls"""
        st.subheader("Pathfinding Options")
        
        # Visualization speed
        st.session_state.visualization_speed = st.slider(
            "Visualization Speed", 0.1, 2.0, 1.0, 0.1
        )
        
        # Show search process
        st.session_state.show_search = st.checkbox(
            "Show Search Process", value=True
        )
        
        # Path optimization
        st.session_state.optimize_path = st.checkbox(
            "Optimize Path", value=True
        )
    
    def main(self):
        """Main application interface"""
        st.title("🧠 Heuristic Pathfinding Dashboard")
        
        # Create tabs for different functionalities
        tab1, tab2, tab3, tab4 = st.tabs([
            "🎮 Interactive Game", 
            "📊 Performance Analysis", 
            "🔄 Algorithm Comparison",
            "📈 Learning Analytics"
        ])
        
        with tab1:
            self.interactive_game_tab()
            
        with tab2:
            self.performance_analysis_tab()
            
        with tab3:
            self.algorithm_comparison_tab()
            
        with tab4:
            self.learning_analytics_tab()
```

### **Supervised Learning Dashboard**

```python
class SupervisedLearningApp(BaseExtensionApp):
    """
    Streamlit dashboard for supervised learning models
    
    Features:
    - Model architecture selection
    - Training progress monitoring
    - Hyperparameter tuning interface
    - Performance evaluation tools
    """
    
    def __init__(self):
        super().__init__("Supervised Learning")
        self.main()
    
    def get_available_algorithms(self) -> List[str]:
        return ["MLP", "CNN", "LSTM", "XGBOOST", "LIGHTGBM", "RANDOMFOREST"]
    
    def add_extension_controls(self):
        """Add ML-specific controls"""
        st.subheader("Training Options")
        
        # Model hyperparameters
        if st.session_state.algorithm in ["MLP", "CNN", "LSTM"]:
            st.session_state.learning_rate = st.number_input(
                "Learning Rate", 0.0001, 0.1, 0.001, format="%.4f"
            )
            st.session_state.batch_size = st.selectbox(
                "Batch Size", [16, 32, 64, 128], index=1
            )
            st.session_state.epochs = st.number_input(
                "Epochs", 10, 1000, 100
            )
        elif st.session_state.algorithm in ["XGBOOST", "LIGHTGBM"]:
            st.session_state.n_estimators = st.number_input(
                "Number of Estimators", 10, 1000, 100
            )
            st.session_state.max_depth = st.number_input(
                "Max Depth", 3, 20, 6
            )
        
        # Dataset selection
        st.session_state.dataset_path = st.selectbox(
            "Training Dataset",
            self.get_available_datasets()
        )
```

## 📊 **Subprocess Integration Patterns**

### **Safe Script Execution**

All Streamlit applications must safely execute backend scripts using standardized patterns:

```python
class SubprocessRunner:
    """
    Safe subprocess execution for Streamlit applications
    
    Design Pattern: Command Pattern
    Purpose: Encapsulate script execution with proper error handling
    Educational Value: Shows how command pattern can manage complex
    operations with undo capability and monitoring.
    """
    
    def __init__(self, extension_path: Path):
        self.extension_path = extension_path
        self.current_process = None
        print(f"[SubprocessRunner] Initialized for {extension_path}")  # Simple logging
        
    def run_script(self, script_name: str, arguments: Dict[str, Any]) -> subprocess.Popen:
        """Execute script with arguments and return process handle"""
        script_path = self.extension_path / "scripts" / script_name
        
        if not script_path.exists():
            raise FileNotFoundError(f"Script not found: {script_path}")
        
        # Build command arguments
        cmd = ["python", str(script_path)]
        for key, value in arguments.items():
            cmd.extend([f"--{key}", str(value)])
        
        # Execute with proper error handling
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=self.extension_path.parent.parent.parent  # Project root
            )
            self.current_process = process
            print(f"[SubprocessRunner] Started process: {' '.join(cmd)}")  # Simple logging
            return process
            
        except Exception as e:
            st.error(f"Failed to start process: {e}")
            raise
```

## 🎯 **User Experience Standards**

### **Responsive Design Principles**
- **Progressive Loading**: Show immediate feedback for user actions
- **Error Recovery**: Graceful handling of subprocess failures
- **State Persistence**: Maintain user preferences across sessions
- **Real-time Updates**: Live progress monitoring for long operations

### **Accessibility Features**
- **Clear Navigation**: Intuitive tab and page organization
- **Helpful Tooltips**: Explanations for technical parameters
- **Visual Feedback**: Progress indicators and status messages
- **Error Messages**: Clear, actionable error descriptions

### **Educational Integration**
- **Algorithm Explanations**: Built-in documentation for each algorithm
- **Parameter Impact**: Visual feedback for parameter changes
- **Performance Insights**: Analytical tools for understanding results
- **Learning Progression**: Guided tutorials and examples

## 📋 **Implementation Checklist**

### **Required Components**
- [ ] **Base Application**: Inherits from standardized base class
- [ ] **Path Management**: Proper working directory and import handling
- [ ] **Algorithm Selection**: Dynamic algorithm factory integration
- [ ] **Parameter Controls**: User-friendly configuration interfaces
- [ ] **Subprocess Integration**: Safe script execution with monitoring
- [ ] **Progress Monitoring**: Real-time feedback for long operations
- [ ] **Error Handling**: Graceful failure recovery and user feedback
- [ ] **Session Management**: Persistent state across user interactions

### **User Interface Requirements**
- [ ] **Multi-tab Layout**: Organized functionality separation
- [ ] **Sidebar Controls**: Consistent parameter adjustment interface
- [ ] **Progress Indicators**: Visual feedback for all operations
- [ ] **Results Display**: Clear visualization of algorithm outputs
- [ ] **Help Documentation**: Integrated explanations and tutorials

### **Integration Requirements**
- [ ] **Factory Pattern**: Dynamic algorithm selection from available implementations
- [ ] **Configuration**: Integration with extension configuration systems
- [ ] **Logging**: Proper integration with project logging infrastructure (simple print statements)
- [ ] **Data Management**: Safe handling of datasets and model files

---

**Streamlit applications represent the user-facing interface of the Snake Game AI project, transforming complex algorithms into accessible, educational, and interactive experiences. They demonstrate how sophisticated technical systems can be made approachable through thoughtful interface design and educational integration.**

## 🔗 **See Also**

- **`scripts.md`**: Script execution and subprocess management
- **`dashboard.md`**: Dashboard architecture and component organization
- **`final-decision-10.md`**: SUPREME_RULES governance system and canonical standards
- **`standalone.md`**: Standalone principle and extension independence
