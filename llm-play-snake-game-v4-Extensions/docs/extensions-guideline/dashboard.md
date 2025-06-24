Each extension folder of v0.03 (not for v0.01, not for v0.02), will have a folder named "dashboard".

For v0.03, it's really important because streamlit app.py will put the streamlit tabs in the dashboard folder to make things more modular.

# Dashboard Architecture for v0.03 Extensions

This document provides comprehensive guidelines for implementing Streamlit dashboards in v0.03 extensions, covering modular design patterns, component architecture, and best practices for interactive data visualization.

## 🎯 **Dashboard Overview**

The `dashboard/` folder is a **mandatory component for all v0.03 extensions**, providing interactive web interfaces for:

- **Real-time monitoring**: Live training/execution progress
- **Performance visualization**: Algorithm comparison and metrics
- **Interactive control**: Parameter tuning and experiment management
- **Data exploration**: Dataset analysis and trajectory visualization
- **Model evaluation**: Comprehensive performance assessment

### **Version-Specific Dashboard Requirements:**

- **v0.01**: ❌ No dashboard (proof-of-concept only)
- **v0.02**: ❌ No dashboard (CLI-focused)
- **v0.03**: ✅ **Mandatory dashboard/** folder with Streamlit components
- **v0.04**: ✅ Enhanced dashboard with JSONL trajectory support (heuristics only)

## 🏗️ **Dashboard Architecture**

### **Standard Dashboard Structure**
```
extensions/[algorithm]-v0.03/
├── dashboard/                    # Mandatory dashboard package
│   ├── __init__.py              # Dashboard initialization and configuration
│   ├── components.py            # Reusable UI components library
│   ├── metrics_buffer.py        # Real-time metrics handling
│   ├── visualization.py         # Advanced visualization components
│   ├── tabs/                    # Modular tab implementations
│   │   ├── __init__.py         # Tab registry and factory
│   │   ├── algorithm_tab.py    # Algorithm execution interface
│   │   ├── comparison_tab.py   # Performance comparison tools
│   │   ├── analysis_tab.py     # Statistical analysis views
│   │   └── datasets_tab.py     # Dataset management interface
│   └── utils/                   # Dashboard utility functions
│       ├── __init__.py         # Utility exports
│       ├── data_processing.py  # Data transformation utilities
│       └── chart_helpers.py    # Chart creation helpers
├── app.py                      # Main Streamlit application entry point
└── proxy_manager.py           # Proxy pattern for v0.02 reuse
```

### **Modular Tab System**
The dashboard implements a **modular tab architecture** where each major functionality is encapsulated in separate tab modules:

```python
# dashboard/tabs/__init__.py
from .algorithm_tab import AlgorithmTab
from .comparison_tab import ComparisonTab
from .analysis_tab import AnalysisTab
from .datasets_tab import DatasetsTab

class TabRegistry:
    """
    Registry for dashboard tabs with dynamic loading
    
    Design Patterns:
    - Registry Pattern: Central tab registration and discovery
    - Factory Pattern: Dynamic tab creation based on configuration
    - Strategy Pattern: Different tab implementations with common interface
    """
    
    _tabs = {}
    
    @classmethod
    def register_tab(cls, tab_name: str, tab_class: type, config: dict = None):
        """Register a tab class for dynamic loading"""
        cls._tabs[tab_name] = {
            'class': tab_class,
            'config': config or {},
            'enabled': True
        }
    
    @classmethod
    def get_tab(cls, tab_name: str):
        """Get tab instance by name"""
        if tab_name not in cls._tabs:
            raise ValueError(f"Tab '{tab_name}' not registered")
        
        tab_info = cls._tabs[tab_name]
        return tab_info['class'](tab_info['config'])
    
    @classmethod
    def get_enabled_tabs(cls) -> dict:
        """Get all enabled tabs"""
        return {name: info for name, info in cls._tabs.items() if info['enabled']}

# Register standard tabs
TabRegistry.register_tab('algorithm', AlgorithmTab, {'icon': '🎯'})
TabRegistry.register_tab('comparison', ComparisonTab, {'icon': '📊'})
TabRegistry.register_tab('analysis', AnalysisTab, {'icon': '📈'})
TabRegistry.register_tab('datasets', DatasetsTab, {'icon': '💾'})
```

## 🎨 **Dashboard Components Library**

### **Reusable Component Architecture**
```python
# dashboard/components.py
import streamlit as st
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import plotly.graph_objects as go
import plotly.express as px

class DashboardComponent(ABC):
    """
    Base class for reusable dashboard components
    
    Design Patterns:
    - Template Method Pattern: Common component lifecycle
    - Observer Pattern: State change notifications
    - Component Pattern: Encapsulated UI elements
    """
    
    def __init__(self, component_id: str, config: Dict = None):
        self.component_id = component_id
        self.config = config or {}
        self.state = {}
        
    @abstractmethod
    def render(self) -> Any:
        """Render the component and return any relevant data"""
        pass
    
    def update_state(self, new_state: Dict):
        """Update component state"""
        self.state.update(new_state)
    
    def get_state(self) -> Dict:
        """Get current component state"""
        return self.state.copy()

class AlgorithmSelectorComponent(DashboardComponent):
    """
    Component for selecting and configuring algorithms
    
    Features:
    - Dynamic algorithm discovery
    - Parameter configuration interface
    - Real-time validation
    - State persistence
    """
    
    def __init__(self, available_algorithms: List[str], config: Dict = None):
        super().__init__("algorithm_selector", config)
        self.available_algorithms = available_algorithms
        
    def render(self) -> Dict[str, Any]:
        """Render algorithm selection interface"""
        
        st.subheader("🤖 Algorithm Selection")
        
        # Algorithm selection
        col1, col2 = st.columns([2, 1])
        
        with col1:
            selected_algorithm = st.selectbox(
                "Choose Algorithm",
                self.available_algorithms,
                key=f"{self.component_id}_algorithm",
                help="Select the algorithm to run or analyze"
            )
        
        with col2:
            algorithm_info = st.button(
                "ℹ️ Info", 
                key=f"{self.component_id}_info",
                help="Show algorithm information"
            )
        
        # Show algorithm information if requested
        if algorithm_info:
            self._show_algorithm_info(selected_algorithm)
        
        # Algorithm parameters
        parameters = self._render_algorithm_parameters(selected_algorithm)
        
        # Update state
        self.state.update({
            'selected_algorithm': selected_algorithm,
            'parameters': parameters
        })
        
        return self.state

    def _render_algorithm_parameters(self, algorithm: str) -> Dict[str, Any]:
        """Render algorithm-specific parameters"""
        
        st.subheader("⚙️ Algorithm Parameters")
        
        parameters = {}
        
        # Get algorithm-specific parameter schema
        param_schema = self._get_parameter_schema(algorithm)
        
        for param_name, param_config in param_schema.items():
            param_type = param_config.get('type', 'float')
            default_value = param_config.get('default')
            min_value = param_config.get('min')
            max_value = param_config.get('max')
            help_text = param_config.get('help', '')
            
            if param_type == 'int':
                parameters[param_name] = st.number_input(
                    param_name.replace('_', ' ').title(),
                    min_value=min_value,
                    max_value=max_value,
                    value=default_value,
                    step=1,
                    help=help_text,
                    key=f"{self.component_id}_{param_name}"
                )
            elif param_type == 'float':
                parameters[param_name] = st.number_input(
                    param_name.replace('_', ' ').title(),
                    min_value=min_value,
                    max_value=max_value,
                    value=default_value,
                    step=0.01,
                    help=help_text,
                    key=f"{self.component_id}_{param_name}"
                )
            elif param_type == 'bool':
                parameters[param_name] = st.checkbox(
                    param_name.replace('_', ' ').title(),
                    value=default_value,
                    help=help_text,
                    key=f"{self.component_id}_{param_name}"
                )
            elif param_type == 'select':
                options = param_config.get('options', [])
                parameters[param_name] = st.selectbox(
                    param_name.replace('_', ' ').title(),
                    options,
                    index=options.index(default_value) if default_value in options else 0,
                    help=help_text,
                    key=f"{self.component_id}_{param_name}"
                )
        
        return parameters

class PerformanceMetricsComponent(DashboardComponent):
    """
    Component for displaying real-time performance metrics
    
    Features:
    - Live metric updates
    - Multiple visualization types
    - Historical data tracking
    - Comparative analysis
    """
    
    def __init__(self, config: Dict = None):
        super().__init__("performance_metrics", config)
        self.metrics_history = []
        
    def render(self, current_metrics: Dict[str, float] = None) -> None:
        """Render performance metrics display"""
        
        st.subheader("📊 Performance Metrics")
        
        if current_metrics:
            self.metrics_history.append(current_metrics)
            
            # Keep only recent metrics
            max_history = self.config.get('max_history', 100)
            if len(self.metrics_history) > max_history:
                self.metrics_history = self.metrics_history[-max_history:]
        
        # Display current metrics
        if current_metrics:
            self._render_current_metrics(current_metrics)
        
        # Display historical trends
        if len(self.metrics_history) > 1:
            self._render_metrics_trends()
    
    def _render_current_metrics(self, metrics: Dict[str, float]):
        """Render current metric values"""
        
        # Create columns for metrics
        num_metrics = len(metrics)
        cols = st.columns(min(num_metrics, 4))
        
        for i, (metric_name, metric_value) in enumerate(metrics.items()):
            with cols[i % 4]:
                # Format metric name
                display_name = metric_name.replace('_', ' ').title()
                
                # Determine metric formatting
                if 'rate' in metric_name.lower() or 'accuracy' in metric_name.lower():
                    # Percentage metrics
                    st.metric(display_name, f"{metric_value:.2%}")
                elif 'loss' in metric_name.lower():
                    # Loss metrics
                    st.metric(display_name, f"{metric_value:.4f}")
                elif 'score' in metric_name.lower():
                    # Score metrics
                    st.metric(display_name, f"{metric_value:.1f}")
                else:
                    # General metrics
                    st.metric(display_name, f"{metric_value:.3f}")
    
    def _render_metrics_trends(self):
        """Render historical metrics trends"""
        
        st.subheader("📈 Metrics Trends")
        
        # Create trend charts
        chart_data = {}
        
        # Extract time series data
        for i, metrics in enumerate(self.metrics_history):
            for metric_name, metric_value in metrics.items():
                if metric_name not in chart_data:
                    chart_data[metric_name] = []
                chart_data[metric_name].append(metric_value)
        
        # Select metrics to display
        available_metrics = list(chart_data.keys())
        selected_metrics = st.multiselect(
            "Select Metrics to Display",
            available_metrics,
            default=available_metrics[:3] if len(available_metrics) > 3 else available_metrics,
            key=f"{self.component_id}_trend_selection"
        )
        
        # Create charts for selected metrics
        for metric_name in selected_metrics:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                y=chart_data[metric_name],
                mode='lines+markers',
                name=metric_name.replace('_', ' ').title(),
                line=dict(width=2)
            ))
            
            fig.update_layout(
                title=f"{metric_name.replace('_', ' ').title()} Over Time",
                xaxis_title="Time Step",
                yaxis_title="Value",
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)

class ProgressTrackerComponent(DashboardComponent):
    """
    Component for tracking long-running operations progress
    
    Features:
    - Multiple progress bars
    - ETA calculations
    - Cancellation support
    - Real-time status updates
    """
    
    def __init__(self, config: Dict = None):
        super().__init__("progress_tracker", config)
        self.active_operations = {}
        
    def start_operation(self, operation_id: str, total_steps: int, description: str = "Processing..."):
        """Start tracking a new operation"""
        
        container = st.container()
        
        with container:
            st.subheader(f"🚀 {description}")
            
            # Progress bar
            progress_bar = st.progress(0)
            
            # Metrics display
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                step_display = st.empty()
            with col2:
                eta_display = st.empty()
            with col3:
                speed_display = st.empty()
            with col4:
                cancel_button = st.button(f"Cancel", key=f"cancel_{operation_id}")
            
            # Status text
            status_text = st.empty()
        
        self.active_operations[operation_id] = {
            'total_steps': total_steps,
            'current_step': 0,
            'start_time': time.time(),
            'progress_bar': progress_bar,
            'step_display': step_display,
            'eta_display': eta_display,
            'speed_display': speed_display,
            'status_text': status_text,
            'cancelled': cancel_button,
            'container': container
        }
        
        return cancel_button
    
    def update_operation(self, operation_id: str, current_step: int, status_message: str = ""):
        """Update operation progress"""
        
        if operation_id not in self.active_operations:
            return
        
        operation = self.active_operations[operation_id]
        progress = current_step / operation['total_steps']
        
        # Update progress bar
        operation['progress_bar'].progress(progress)
        
        # Update step counter
        operation['step_display'].metric(
            "Progress",
            f"{current_step}/{operation['total_steps']}"
        )
        
        # Calculate ETA
        elapsed_time = time.time() - operation['start_time']
        if current_step > 0:
            time_per_step = elapsed_time / current_step
            remaining_steps = operation['total_steps'] - current_step
            eta_seconds = time_per_step * remaining_steps
            
            operation['eta_display'].metric(
                "ETA",
                f"{eta_seconds/60:.1f}m"
            )
            
            operation['speed_display'].metric(
                "Speed",
                f"{1/time_per_step:.1f} steps/s"
            )
        
        # Update status
        if status_message:
            operation['status_text'].text(status_message)
        
        operation['current_step'] = current_step
    
    def is_cancelled(self, operation_id: str) -> bool:
        """Check if operation was cancelled"""
        if operation_id not in self.active_operations:
            return False
        
        return self.active_operations[operation_id]['cancelled']
    
    def complete_operation(self, operation_id: str, success_message: str = "Completed!"):
        """Mark operation as complete"""
        if operation_id in self.active_operations:
            operation = self.active_operations[operation_id]
            operation['progress_bar'].progress(1.0)
            operation['status_text'].success(success_message)
            
            # Clean up after a delay
            time.sleep(2)
            del self.active_operations[operation_id]
```

## 📊 **Real-Time Metrics Management**

### **Metrics Buffer System**
```python
# dashboard/metrics_buffer.py
import threading
import queue
import time
from typing import Dict, List, Optional, Callable
from collections import deque, defaultdict

class MetricsBuffer:
    """
    Thread-safe buffer for real-time metrics collection and distribution
    
    Features:
    - Multi-producer, multi-consumer pattern
    - Automatic data aggregation
    - Time-windowed metrics
    - Memory-efficient circular buffers
    - Event-driven updates
    """
    
    def __init__(self, buffer_size: int = 1000, update_interval: float = 1.0):
        self.buffer_size = buffer_size
        self.update_interval = update_interval
        
        # Thread-safe data structures
        self._metrics_queue = queue.Queue()
        self._metrics_buffer = defaultdict(lambda: deque(maxlen=buffer_size))
        self._subscribers = []
        self._lock = threading.Lock()
        
        # Background processing
        self._running = True
        self._worker_thread = threading.Thread(target=self._process_metrics, daemon=True)
        self._worker_thread.start()
    
    def add_metrics(self, metrics: Dict[str, float], timestamp: Optional[float] = None):
        """Add metrics to the buffer"""
        if timestamp is None:
            timestamp = time.time()
        
        self._metrics_queue.put({
            'metrics': metrics,
            'timestamp': timestamp
        })
    
    def subscribe(self, callback: Callable[[Dict], None]):
        """Subscribe to metrics updates"""
        with self._lock:
            self._subscribers.append(callback)
    
    def unsubscribe(self, callback: Callable[[Dict], None]):
        """Unsubscribe from metrics updates"""
        with self._lock:
            if callback in self._subscribers:
                self._subscribers.remove(callback)
    
    def get_latest_metrics(self, metric_names: Optional[List[str]] = None) -> Dict[str, float]:
        """Get latest values for specified metrics"""
        with self._lock:
            latest_metrics = {}
            
            target_metrics = metric_names or list(self._metrics_buffer.keys())
            
            for metric_name in target_metrics:
                if metric_name in self._metrics_buffer and self._metrics_buffer[metric_name]:
                    latest_metrics[metric_name] = self._metrics_buffer[metric_name][-1]['value']
            
            return latest_metrics
    
    def get_metrics_history(self, metric_name: str, window_size: Optional[int] = None) -> List[Dict]:
        """Get historical data for a specific metric"""
        with self._lock:
            if metric_name not in self._metrics_buffer:
                return []
            
            history = list(self._metrics_buffer[metric_name])
            
            if window_size:
                history = history[-window_size:]
            
            return history
    
    def get_aggregated_metrics(self, window_seconds: float = 60.0) -> Dict[str, Dict[str, float]]:
        """Get aggregated metrics for specified time window"""
        current_time = time.time()
        cutoff_time = current_time - window_seconds
        
        aggregated = {}
        
        with self._lock:
            for metric_name, metric_buffer in self._metrics_buffer.items():
                # Filter metrics within time window
                recent_metrics = [
                    item for item in metric_buffer 
                    if item['timestamp'] >= cutoff_time
                ]
                
                if recent_metrics:
                    values = [item['value'] for item in recent_metrics]
                    
                    aggregated[metric_name] = {
                        'mean': sum(values) / len(values),
                        'min': min(values),
                        'max': max(values),
                        'count': len(values),
                        'latest': values[-1]
                    }
        
        return aggregated
    
    def _process_metrics(self):
        """Background thread for processing metrics queue"""
        while self._running:
            try:
                # Process queued metrics
                while not self._metrics_queue.empty():
                    metrics_data = self._metrics_queue.get_nowait()
                    self._store_metrics(metrics_data)
                
                # Notify subscribers
                self._notify_subscribers()
                
                # Sleep until next update
                time.sleep(self.update_interval)
                
            except Exception as e:
                print(f"Error in metrics processing: {e}")
    
    def _store_metrics(self, metrics_data: Dict):
        """Store metrics in buffer"""
        with self._lock:
            timestamp = metrics_data['timestamp']
            
            for metric_name, metric_value in metrics_data['metrics'].items():
                self._metrics_buffer[metric_name].append({
                    'value': metric_value,
                    'timestamp': timestamp
                })
    
    def _notify_subscribers(self):
        """Notify all subscribers of metric updates"""
        with self._lock:
            if self._subscribers:
                latest_metrics = self.get_latest_metrics()
                
                for callback in self._subscribers[:]:  # Copy list to avoid modification during iteration
                    try:
                        callback(latest_metrics)
                    except Exception as e:
                        print(f"Error notifying subscriber: {e}")
    
    def shutdown(self):
        """Shutdown the metrics buffer"""
        self._running = False
        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5.0)
```

## 🎮 **Tab Implementation Examples**

### **Algorithm Execution Tab**
```python
# dashboard/tabs/algorithm_tab.py
import streamlit as st
from typing import Dict, Any
from ..components import AlgorithmSelectorComponent, ProgressTrackerComponent
from ..metrics_buffer import MetricsBuffer

class AlgorithmTab:
    """
    Tab for algorithm execution and real-time monitoring
    
    Features:
    - Algorithm selection and configuration
    - Real-time execution monitoring
    - Live performance metrics
    - Interactive control and cancellation
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.metrics_buffer = MetricsBuffer()
        
        # Initialize components
        available_algorithms = self._get_available_algorithms()
        self.algorithm_selector = AlgorithmSelectorComponent(available_algorithms)
        self.progress_tracker = ProgressTrackerComponent()
    
    def render(self):
        """Render the algorithm execution tab"""
        
        st.header("🎯 Algorithm Execution")
        
        # Algorithm selection section
        algorithm_config = self.algorithm_selector.render()
        
        st.divider()
        
        # Execution controls
        self._render_execution_controls(algorithm_config)
        
        st.divider()
        
        # Real-time monitoring
        self._render_real_time_monitoring()
    
    def _render_execution_controls(self, algorithm_config: Dict):
        """Render execution control interface"""
        
        st.subheader("🎮 Execution Controls")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            num_games = st.number_input(
                "Number of Games",
                min_value=1,
                max_value=1000,
                value=10,
                help="Number of games to run"
            )
        
        with col2:
            grid_size = st.selectbox(
                "Grid Size",
                [8, 10, 12, 16, 20],
                index=1,
                help="Size of the game grid"
            )
        
        with col3:
            visualization = st.checkbox(
                "Enable Visualization",
                value=False,
                help="Show game visualization (slower)"
            )
        
        # Start execution button
        if st.button("🚀 Start Execution", type="primary"):
            self._start_algorithm_execution(
                algorithm_config,
                num_games,
                grid_size,
                visualization
            )
    
    def _render_real_time_monitoring(self):
        """Render real-time monitoring section"""
        
        st.subheader("📊 Real-Time Monitoring")
        
        # Check for active executions
        if 'active_execution' in st.session_state:
            # Show progress and metrics
            self._show_execution_progress()
        else:
            st.info("No active execution. Start an algorithm to see real-time metrics.")
    
    def _start_algorithm_execution(self, algorithm_config: Dict, num_games: int, 
                                 grid_size: int, visualization: bool):
        """Start algorithm execution with monitoring"""
        
        # Store execution state
        st.session_state.active_execution = {
            'algorithm': algorithm_config['selected_algorithm'],
            'parameters': algorithm_config['parameters'],
            'num_games': num_games,
            'grid_size': grid_size,
            'visualization': visualization,
            'current_game': 0
        }
        
        # Start progress tracking
        operation_id = f"algorithm_execution_{int(time.time())}"
        cancel_button = self.progress_tracker.start_operation(
            operation_id,
            num_games,
            f"Running {algorithm_config['selected_algorithm']} Algorithm"
        )
        
        st.session_state.execution_operation_id = operation_id
        
        # Trigger rerun to update display
        st.experimental_rerun()
```

## 🚀 **Main Application Integration**

### **app.py Structure**
```python
# app.py
import streamlit as st
from dashboard.tabs import TabRegistry
from dashboard.metrics_buffer import MetricsBuffer

def main():
    """Main Streamlit application for v0.03 extension"""
    
    # Page configuration
    st.set_page_config(
        page_title="Snake Game AI - [Extension] v0.03",
        page_icon="🐍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Main title
    st.title("🐍 Snake Game AI - [Extension] v0.03 Dashboard")
    
    # Initialize metrics buffer
    if 'metrics_buffer' not in st.session_state:
        st.session_state.metrics_buffer = MetricsBuffer()
    
    # Get enabled tabs
    enabled_tabs = TabRegistry.get_enabled_tabs()
    
    # Create tab navigation
    tab_names = []
    tab_icons = []
    
    for tab_name, tab_info in enabled_tabs.items():
        icon = tab_info['config'].get('icon', '📊')
        display_name = tab_name.replace('_', ' ').title()
        tab_names.append(display_name)
        tab_icons.append(f"{icon} {display_name}")
    
    # Render tabs
    tabs = st.tabs(tab_icons)
    
    for i, (tab_name, tab_info) in enumerate(enabled_tabs.items()):
        with tabs[i]:
            tab_instance = TabRegistry.get_tab(tab_name)
            tab_instance.render()

if __name__ == "__main__":
    main()
```

## 🎯 **Best Practices for Dashboard Development**

### **1. Performance Optimization**
- Use `st.cache_data` for expensive computations
- Implement lazy loading for large datasets
- Use session state for persistent data
- Optimize chart rendering with sampling for large datasets

### **2. User Experience**
- Provide clear progress indicators for long operations
- Include helpful tooltips and documentation
- Implement graceful error handling and recovery
- Ensure responsive design for different screen sizes

### **3. Code Organization**
- Keep components modular and reusable
- Use consistent naming conventions
- Implement proper separation of concerns
- Document component APIs and usage patterns

### **4. Testing and Validation**
- Test components in isolation
- Validate user inputs and provide feedback
- Handle edge cases gracefully
- Implement proper error logging

---

**The dashboard architecture provides a comprehensive, modular framework for creating rich interactive interfaces that enhance the user experience and provide powerful tools for algorithm analysis and experimentation across all v0.03 extensions.**



