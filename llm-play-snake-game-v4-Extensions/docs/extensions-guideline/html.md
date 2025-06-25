# HTML Template Architecture for Extensions

> **Important — Authoritative Reference:** This document supplements the _Final Decision Series_ and extension guidelines. HTML templates follow the same architectural patterns established in the GOODRULES.

## 🎯 **Core Philosophy: Generic Templates, Dynamic Content**

The HTML template architecture demonstrates perfect template inheritance patterns where generic base templates provide foundation structure while extension-specific content is injected dynamically, eliminating the need for custom HTML files in extensions.

### **Design Philosophy**
- **Template Reusability**: Generic templates work for all extensions
- **Dynamic Content Injection**: Context variables customize template behavior
- **Template Inheritance**: Hierarchical template structure with extension points
- **Framework Agnostic**: Compatible with Flask, Django, and other web frameworks

## 🏗️ **Template Architecture**

### **Existing Generic Templates**
The current template structure in `web/templates/` is already designed for universal reuse:

```html
<!-- web/templates/main.html - Universal game interface -->
<!DOCTYPE html>
<html>
<head>
    <title>{{ title|default("Snake Game AI") }}</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='css/style.css') }}">
</head>
<body>
    <!-- Universal game board -->
    <div id="game-board"></div>
    
    <!-- Dynamic mode-specific content -->
    {% if show_llm_response %}
        <div id="llm-section">{{ llm_response }}</div>
    {% elif show_algorithm_info %}
        <div id="algorithm-section">{{ algorithm_info }}</div>
    {% elif show_training_metrics %}
        <div id="training-section">{{ training_metrics }}</div>
    {% endif %}
    
    <!-- Universal controls -->
    <div id="controls">
        <button id="start-btn">Start</button>
        <button id="pause-btn">Pause</button>
        <button id="reset-btn">Reset</button>
    </div>
    
    <script src="{{ url_for('static', filename='js/main.js') }}"></script>
</body>
</html>
```

### **Template Inheritance Strategy**
Following template method patterns for extensibility:

```html
<!-- extensions/heuristics-v0.03/templates/heuristic.html -->
{% extends "main.html" %}

{% block title %}Snake Game – Heuristic Mode{% endblock %}

{% block mode_specific_content %}
<div class="algorithm-section">
    <h2>Algorithm Information</h2>
    <div>Current: <span id="algorithm-name">{{ algorithm_name }}</span></div>
    <div>Path Length: <span id="path-length">{{ path_length }}</span></div>
    <div>Nodes Explored: <span id="nodes-explored">{{ nodes_explored }}</span></div>
</div>
{% endblock %}

{% block additional_scripts %}
<script src="{{ url_for('static', filename='js/heuristic_visualization.js') }}"></script>
{% endblock %}
```

## 🔧 **Dynamic Content Strategy**

### **Context Variable Injection**
Extensions customize templates through controller context:

```python
# Heuristic Game Controller
class HeuristicWebController(BaseWebController):
    """Web controller for heuristic algorithms"""
    
    def get_template_context(self):
        """Provide heuristic-specific template context"""
        return {
            "title": "Snake Game – Heuristic Pathfinding",
            "show_algorithm_info": True,
            "show_llm_response": False,
            "algorithm_name": self.current_algorithm,
            "mode_specific_data": {
                "path_length": len(self.pathfinder.current_path),
                "nodes_explored": self.pathfinder.stats.nodes_explored,
                "search_time": self.pathfinder.stats.search_time
            }
        }

# Supervised Learning Controller  
class SupervisedWebController(BaseWebController):
    """Web controller for supervised learning models"""
    
    def get_template_context(self):
        """Provide ML-specific template context"""
        return {
            "title": "Snake Game – ML Model Evaluation",
            "show_training_metrics": True,
            "show_llm_response": False,
            "model_name": self.current_model,
            "mode_specific_data": {
                "prediction_confidence": self.model.last_confidence,
                "model_accuracy": self.model.validation_accuracy,
                "inference_time": self.model.last_inference_time
            }
        }
```

### **JavaScript Adaptability**
Generic JavaScript handles different data types seamlessly:

```javascript
// web/static/js/main.js - Universal game interface
class GameInterface {
    constructor() {
        this.mode = this.detectMode();
        this.setupEventHandlers();
    }
    
    updateGameInfo(data) {
        // Universal game state updates
        document.getElementById('score').textContent = data.score || 0;
        document.getElementById('steps').textContent = data.steps || 0;
        
        // Mode-specific content updates
        if (data.llm_response) {
            // Task-0: LLM response display
            this.updateLLMInfo(data.llm_response);
        } else if (data.algorithm_info) {
            // Heuristics: Algorithm information
            this.updateAlgorithmInfo(data.algorithm_info);
        } else if (data.training_metrics) {
            // ML: Training/prediction metrics
            this.updateMLInfo(data.training_metrics);
        } else if (data.rl_metrics) {
            // RL: Training progress and Q-values
            this.updateRLInfo(data.rl_metrics);
        }
    }
    
    updateAlgorithmInfo(info) {
        document.getElementById('algorithm-name').textContent = info.name;
        document.getElementById('path-length').textContent = info.path_length;
        document.getElementById('nodes-explored').textContent = info.nodes_explored;
    }
    
    updateMLInfo(metrics) {
        document.getElementById('model-name').textContent = metrics.model_name;
        document.getElementById('confidence').textContent = 
            `${(metrics.confidence * 100).toFixed(1)}%`;
        document.getElementById('inference-time').textContent = 
            `${metrics.inference_time.toFixed(3)}ms`;
    }
}
```

## 🎨 **CSS Architecture**

### **Generic Styling Classes**
CSS uses semantic classes that work across all extensions:

```css
/* web/static/css/style.css - Universal styling */
.game-container, .stats-section, .controls-section {
    /* Layout styles work for all modes */
}

.algorithm-section, .llm-section, .training-section, .rl-section {
    /* Mode-specific sections with consistent styling */
}

.metric-display, .info-panel, .status-indicator {
    /* Reusable component styles */
}
```

## 🚀 **Extension Benefits**

### **No HTML Development Required**
Extensions only need to:
1. **Provide context data** through their web controllers
2. **Update JavaScript** to handle their specific data formats  
3. **Optionally extend** templates for specialized content
4. **Use existing** CSS classes for consistent styling

### **Consistent User Experience**
- **Uniform Layout**: Same visual structure across all extensions
- **Familiar Controls**: Consistent button placement and behavior
- **Responsive Design**: Works across desktop and mobile devices
- **Accessible Interface**: Standard HTML elements with proper semantics

### **Rapid Development**
- **No Template Duplication**: Reuse existing template infrastructure
- **Minimal Customization**: Only extension-specific content needs attention
- **Quick Iteration**: Changes to base templates benefit all extensions
- **Easy Maintenance**: Centralized template logic reduces bugs

## 🔧 **Advanced Template Patterns**

### **Component-Based Architecture**
```html
<!-- Reusable template components -->
{% macro render_game_stats(stats) %}
<div class="stats-panel">
    <div class="stat-item">Score: {{ stats.score }}</div>
    <div class="stat-item">Steps: {{ stats.steps }}</div>
    <div class="stat-item">Time: {{ stats.duration }}</div>
</div>
{% endmacro %}

{% macro render_algorithm_panel(algorithm_data) %}
<div class="algorithm-panel">
    <h3>{{ algorithm_data.name }}</h3>
    <div class="algorithm-metrics">
        {% for key, value in algorithm_data.metrics.items() %}
            <div class="metric">{{ key }}: {{ value }}</div>
        {% endfor %}
    </div>
</div>
{% endmacro %}
```

### **Progressive Enhancement**
```javascript
// Progressive enhancement for advanced features
if (window.WebSocket && window.requestAnimationFrame) {
    // Real-time updates for supported browsers
    this.enableRealtimeUpdates();
} else {
    // Fallback polling for older browsers
    this.enablePollingUpdates();
}
```

---

**The HTML template architecture eliminates the need for custom HTML development in extensions by providing a robust, flexible foundation that adapts to different content types through dynamic context injection and template inheritance, maintaining consistency while enabling specialization.**


