# Fine-Tuning LLMs for Snake Game AI

> **Reference:** See `final-decision-10.md` (SUPREME_RULES), `heuristics-as-foundation.md`, `llm-distillation.md`, `data-format-decision-guide.md`.

## 🎯 Purpose

Fine-tuning LLMs for Snake Game AI uses high-quality heuristic datasets to teach reasoning and structured problem-solving, following SUPREME_RULES from `final-decision-10.md`.

## 🏗️ Canonical Fine-Tuning Pattern

```python
from utils.factory_utils import SimpleFactory
from utils.print_utils import print_info

class LLMFineTuningFactory:
    _registry = {
        "OPENAI_GPT": OpenAIFineTuner,
        "HUGGINGFACE": HuggingFaceFineTuner,
        "LOCAL_MODEL": LocalFineTuner,
    }
    @classmethod
    def create(cls, model_type: str, **kwargs):
        tuner_class = cls._registry.get(model_type.upper())
        if not tuner_class:
            raise ValueError(f"Unknown model type: {model_type}")
        print_info(f"[LLMFineTuningFactory] Creating tuner: {model_type}")
        return tuner_class(**kwargs)

# Data preparation and training
class FineTuningDataProcessor:
    def __init__(self):
        print_info("[FineTuningDataProcessor] Initialized")
    def prepare_training_data(self, heuristic_datasets: list) -> list:
        # ... load and convert data ...
        print_info(f"[FineTuningDataProcessor] Generated {len(heuristic_datasets)} training examples")
        return []

def fine_tune_model(model_config: dict, training_data: list):
    print_info(f"[FineTuner] Starting fine-tuning with {len(training_data)} examples")
    # ... training loop ...
    print_info(f"[FineTuner] Fine-tuning completed")
```

## ✅ Key Points
- Use canonical `create()` method for all factories.
- Use print_utils for all logging.
- Data source should be heuristics-v0.04 JSONL。
- No .log files, no complex logging.


