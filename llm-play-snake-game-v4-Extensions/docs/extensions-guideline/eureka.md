# Eureka: Reward Function Evolution for Snake Game AI

This document outlines the implementation of Eureka-style automated reward function generation and evolution for reinforcement learning agents in the Snake Game AI project.

## 🎯 **Overview**

Eureka represents a paradigm shift in reinforcement learning where reward functions are automatically generated and evolved using large language models (LLMs) rather than being hand-crafted by humans. This approach can discover novel and effective reward shaping strategies that might not be intuitive to human designers.

### **Core Concept**
- **Traditional RL**: Human-designed reward functions
- **Eureka Approach**: LLM-generated and evolved reward functions
- **Benefits**: Discovers non-obvious reward patterns, reduces human bias, enables rapid experimentation

## 🧬 **Eureka Architecture for Snake Game**

### **Extension Structure** # TODO: this is not the final structure. Maybe it's good, maybe not. Up to you to adopt it or not.
```
extensions/eureka-v0.01/
├── __init__.py
├── eureka_engine.py          # Main Eureka evolution engine
├── reward_generator.py       # LLM-based reward function generation
├── reward_evaluator.py       # Reward function evaluation framework
├── code_executor.py          # Safe reward function execution
├── population_manager.py     # Reward function population management
├── llm_interface.py          # LOCAL LLM API integration, OLLAMA
└── templates/
    ├── reward_templates.py   # Base reward function templates
    └── prompts/              # LLM prompts for generation
```

---

**Eureka represents a revolutionary approach to reward function design, enabling the discovery of novel and effective strategies through automated LLM-based evolution. This approach can uncover reward patterns that human designers might never consider, leading to breakthrough performance in Snake Game AI.**

