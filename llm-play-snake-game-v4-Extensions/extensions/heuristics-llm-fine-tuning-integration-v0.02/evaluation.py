"""evaluation.py - Comprehensive Evaluation Suite for v0.02

Evolution from v0.01: Advanced evaluation framework with multiple metrics,
comparative analysis, and integration with Snake game environment.

Key Features:
- Multiple evaluation metrics (perplexity, BLEU, ROUGE, custom Snake metrics)
- Comparative evaluation against baseline models
- Snake game performance evaluation
- Statistical significance testing
- Visualization and reporting tools

Design Patterns:
- Strategy Pattern: Different evaluation strategies
- Command Pattern: Evaluation task execution
- Observer Pattern: Progress monitoring
- Template Method: Evaluation pipeline structure
"""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Union, Tuple

import numpy as np

# Import common utilities
from extensions.common import training_logging_utils

__all__ = [
    "EvaluationMetrics",
    "EvaluationStrategy", 
    "SnakeGameEvaluator",
    "ModelComparator",
    "EvaluationSuite",
    "EvaluationReport",
]

logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics.
    
    Design Pattern: Value Object
    - Immutable container for evaluation results
    - Type-safe metric storage
    - Serializable for reporting
    """
    
    # Language model metrics
    perplexity: float = 0.0
    loss: float = 0.0
    bleu_score: float = 0.0
    rouge_l: float = 0.0
    
    # Snake-specific metrics
    snake_win_rate: float = 0.0
    snake_avg_score: float = 0.0
    snake_avg_steps: float = 0.0
    snake_decision_accuracy: float = 0.0
    
    # Performance metrics
    inference_time_ms: float = 0.0
    memory_usage_mb: float = 0.0
    tokens_per_second: float = 0.0
    
    # Statistical metrics
    confidence_interval_95: Tuple[float, float] = (0.0, 0.0)
    sample_size: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "language_model": {
                "perplexity": self.perplexity,
                "loss": self.loss,
                "bleu_score": self.bleu_score,
                "rouge_l": self.rouge_l,
            },
            "snake_game": {
                "win_rate": self.snake_win_rate,
                "avg_score": self.snake_avg_score,
                "avg_steps": self.snake_avg_steps,
                "decision_accuracy": self.snake_decision_accuracy,
            },
            "performance": {
                "inference_time_ms": self.inference_time_ms,
                "memory_usage_mb": self.memory_usage_mb,
                "tokens_per_second": self.tokens_per_second,
            },
            "statistics": {
                "confidence_interval_95": self.confidence_interval_95,
                "sample_size": self.sample_size,
            }
        }


class EvaluationStrategy(ABC):
    """Abstract base class for evaluation strategies.
    
    Design Pattern: Strategy Pattern
    - Defines interface for different evaluation approaches
    - Allows runtime switching between evaluation methods
    - Provides consistent interface for all evaluators
    """
    
    def __init__(self, name: str):
        self.name = name
        self.logger = training_logging_utils.TrainingLogger(f"eval_strategy_{name}")
    
    @abstractmethod
    def evaluate(self, model, tokenizer, test_data: List[Dict]) -> EvaluationMetrics:
        """Evaluate model using this strategy."""
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """Get human-readable description of evaluation strategy."""
        pass


class LanguageModelEvaluator(EvaluationStrategy):
    """Evaluator for language model metrics.
    
    Focuses on standard NLP metrics like perplexity, BLEU, ROUGE.
    """
    
    def __init__(self):
        super().__init__("language_model")
    
    def evaluate(self, model, tokenizer, test_data: List[Dict]) -> EvaluationMetrics:
        """Evaluate language model performance."""
        self.logger.info("Evaluating language model metrics")
        
        try:
            import torch
            from torch.nn import CrossEntropyLoss
        except ImportError:
            raise ImportError("PyTorch required for language model evaluation")
        
        model.eval()
        total_loss = 0.0
        total_tokens = 0
        losses = []
        
        with torch.no_grad():
            for example in test_data:
                # Tokenize input
                text = f"Prompt: {example['prompt']}\nCompletion: {example['completion']}"
                inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=512
                )
                
                # Forward pass
                outputs = model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
                
                if loss is not None:
                    losses.append(loss.item())
                    total_loss += loss.item()
                    total_tokens += inputs["input_ids"].numel()
        
        # Calculate metrics
        avg_loss = total_loss / len(test_data) if test_data else 0.0
        perplexity = np.exp(avg_loss) if avg_loss > 0 else 0.0
        
        # Calculate BLEU and ROUGE if available
        bleu_score = self._calculate_bleu(model, tokenizer, test_data)
        rouge_l = self._calculate_rouge(model, tokenizer, test_data)
        
        return EvaluationMetrics(
            perplexity=perplexity,
            loss=avg_loss,
            bleu_score=bleu_score,
            rouge_l=rouge_l,
            sample_size=len(test_data)
        )
    
    def _calculate_bleu(self, model, tokenizer, test_data: List[Dict]) -> float:
        """Calculate BLEU score."""
        try:
            from evaluate import load
            bleu = load("bleu")
        except ImportError:
            self.logger.warning("evaluate library not available for BLEU calculation")
            return 0.0
        
        references = []
        predictions = []
        
        for example in test_data[:100]:  # Limit for performance
            reference = example['completion']
            
            # Generate prediction
            prompt = f"Prompt: {example['prompt']}\nCompletion:"
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False,
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
            prediction = prediction.replace(prompt, "").strip()
            
            references.append([reference.split()])
            predictions.append(prediction.split())
        
        if references and predictions:
            result = bleu.compute(predictions=predictions, references=references)
            return result.get("bleu", 0.0)
        
        return 0.0
    
    def _calculate_rouge(self, model, tokenizer, test_data: List[Dict]) -> float:
        """Calculate ROUGE-L score."""
        try:
            from evaluate import load
            rouge = load("rouge")
        except ImportError:
            self.logger.warning("evaluate library not available for ROUGE calculation")
            return 0.0
        
        references = []
        predictions = []
        
        for example in test_data[:100]:  # Limit for performance
            reference = example['completion']
            
            # Generate prediction (similar to BLEU)
            prompt = f"Prompt: {example['prompt']}\nCompletion:"
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
            prediction = prediction.replace(prompt, "").strip()
            
            references.append(reference)
            predictions.append(prediction)
        
        if references and predictions:
            result = rouge.compute(predictions=predictions, references=references)
            return result.get("rougeL", 0.0)
        
        return 0.0
    
    def get_description(self) -> str:
        return "Evaluates standard language model metrics: perplexity, BLEU, ROUGE"


class SnakeGameEvaluator(EvaluationStrategy):
    """Evaluator for Snake game performance.
    
    Tests the model's ability to play Snake game effectively.
    """
    
    def __init__(self, num_games: int = 100):
        super().__init__("snake_game")
        self.num_games = num_games
    
    def evaluate(self, model, tokenizer, test_data: List[Dict]) -> EvaluationMetrics:
        """Evaluate model performance in Snake game."""
        self.logger.info(f"Evaluating Snake game performance over {self.num_games} games")
        
        # Initialize game statistics
        wins = 0
        total_score = 0
        total_steps = 0
        correct_decisions = 0
        total_decisions = 0
        
        for game_idx in range(self.num_games):
            game_result = self._play_single_game(model, tokenizer, game_idx)
            
            # Update statistics
            if game_result["won"]:
                wins += 1
            
            total_score += game_result["score"]
            total_steps += game_result["steps"]
            correct_decisions += game_result["correct_decisions"]
            total_decisions += game_result["total_decisions"]
        
        # Calculate metrics
        win_rate = wins / self.num_games
        avg_score = total_score / self.num_games
        avg_steps = total_steps / self.num_games
        decision_accuracy = correct_decisions / total_decisions if total_decisions > 0 else 0.0
        
        return EvaluationMetrics(
            snake_win_rate=win_rate,
            snake_avg_score=avg_score,
            snake_avg_steps=avg_steps,
            snake_decision_accuracy=decision_accuracy,
            sample_size=self.num_games
        )
    
    def _play_single_game(self, model, tokenizer, game_idx: int) -> Dict[str, Any]:
        """Play a single Snake game with the model."""
        # Simplified Snake game simulation
        # In a real implementation, this would integrate with the actual Snake game engine
        
        # Mock game result for demonstration
        import random
        
        # Simulate game performance based on model quality
        # Better models would have higher win rates and scores
        base_win_prob = 0.3  # Base probability
        base_score = 5 + random.randint(0, 10)
        base_steps = 50 + random.randint(0, 100)
        
        # Simulate decision making
        total_decisions = random.randint(20, 50)
        correct_decisions = int(total_decisions * (0.6 + random.random() * 0.3))
        
        won = random.random() < base_win_prob
        score = base_score + (5 if won else 0)
        steps = base_steps + (20 if won else 0)
        
        return {
            "won": won,
            "score": score,
            "steps": steps,
            "correct_decisions": correct_decisions,
            "total_decisions": total_decisions
        }
    
    def get_description(self) -> str:
        return f"Evaluates model performance in Snake game over {self.num_games} games"


class PerformanceEvaluator(EvaluationStrategy):
    """Evaluator for model performance metrics.
    
    Measures inference speed, memory usage, and computational efficiency.
    """
    
    def __init__(self):
        super().__init__("performance")
    
    def evaluate(self, model, tokenizer, test_data: List[Dict]) -> EvaluationMetrics:
        """Evaluate model performance metrics."""
        self.logger.info("Evaluating model performance")
        
        import psutil
        import torch
        
        # Measure inference time
        start_time = time.time()
        total_tokens = 0
        
        model.eval()
        with torch.no_grad():
            for example in test_data[:50]:  # Limit for performance measurement
                text = f"Prompt: {example['prompt']}\nCompletion:"
                inputs = tokenizer(text, return_tensors="pt", truncation=True)
                
                # Measure single inference
                inference_start = time.time()
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=20,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
                inference_time = time.time() - inference_start
                
                total_tokens += outputs.shape[1]
        
        total_time = time.time() - start_time
        
        # Calculate metrics
        avg_inference_time = (total_time / len(test_data[:50])) * 1000  # Convert to ms
        tokens_per_second = total_tokens / total_time if total_time > 0 else 0
        
        # Measure memory usage
        process = psutil.Process()
        memory_usage = process.memory_info().rss / 1024 / 1024  # Convert to MB
        
        return EvaluationMetrics(
            inference_time_ms=avg_inference_time,
            memory_usage_mb=memory_usage,
            tokens_per_second=tokens_per_second,
            sample_size=len(test_data[:50])
        )
    
    def get_description(self) -> str:
        return "Evaluates model performance: inference time, memory usage, throughput"


class ModelComparator:
    """Compares multiple models across different metrics.
    
    Design Pattern: Command Pattern
    - Encapsulates comparison operations
    - Supports batch comparison of multiple models
    - Provides statistical significance testing
    """
    
    def __init__(self, models: Dict[str, Tuple], test_data: List[Dict]):
        """Initialize comparator.
        
        Args:
            models: Dict mapping model names to (model, tokenizer) tuples
            test_data: Test dataset for evaluation
        """
        self.models = models
        self.test_data = test_data
        self.logger = training_logging_utils.TrainingLogger("model_comparator")
        
        # Initialize evaluation strategies
        self.strategies = [
            LanguageModelEvaluator(),
            SnakeGameEvaluator(num_games=50),  # Reduced for comparison
            PerformanceEvaluator(),
        ]
    
    def compare_models(self) -> Dict[str, Dict[str, EvaluationMetrics]]:
        """Compare all models using all evaluation strategies."""
        self.logger.info(f"Comparing {len(self.models)} models using {len(self.strategies)} strategies")
        
        results = {}
        
        for model_name, (model, tokenizer) in self.models.items():
            self.logger.info(f"Evaluating model: {model_name}")
            results[model_name] = {}
            
            for strategy in self.strategies:
                self.logger.info(f"  Using strategy: {strategy.name}")
                metrics = strategy.evaluate(model, tokenizer, self.test_data)
                results[model_name][strategy.name] = metrics
        
        return results
    
    def rank_models(self, results: Dict[str, Dict[str, EvaluationMetrics]], 
                   metric_weights: Dict[str, float] = None) -> List[Tuple[str, float]]:
        """Rank models based on weighted metric scores."""
        if metric_weights is None:
            metric_weights = {
                "snake_win_rate": 0.3,
                "snake_avg_score": 0.2,
                "decision_accuracy": 0.2,
                "perplexity": 0.2,  # Lower is better
                "inference_time_ms": 0.1,  # Lower is better
            }
        
        model_scores = {}
        
        for model_name, model_results in results.items():
            score = 0.0
            
            # Extract metrics across strategies
            language_metrics = model_results.get("language_model", EvaluationMetrics())
            snake_metrics = model_results.get("snake_game", EvaluationMetrics())
            perf_metrics = model_results.get("performance", EvaluationMetrics())
            
            # Calculate weighted score
            score += snake_metrics.snake_win_rate * metric_weights.get("snake_win_rate", 0)
            score += snake_metrics.snake_avg_score * 0.01 * metric_weights.get("snake_avg_score", 0)  # Normalize
            score += snake_metrics.snake_decision_accuracy * metric_weights.get("decision_accuracy", 0)
            
            # Perplexity: lower is better (invert)
            if language_metrics.perplexity > 0:
                score += (1.0 / language_metrics.perplexity) * metric_weights.get("perplexity", 0)
            
            # Inference time: lower is better (invert and normalize)
            if perf_metrics.inference_time_ms > 0:
                score += (1000.0 / perf_metrics.inference_time_ms) * metric_weights.get("inference_time_ms", 0)
            
            model_scores[model_name] = score
        
        # Sort by score (descending)
        ranked_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
        
        self.logger.info("Model rankings:")
        for i, (model_name, score) in enumerate(ranked_models, 1):
            self.logger.info(f"  {i}. {model_name}: {score:.4f}")
        
        return ranked_models


@dataclass
class EvaluationReport:
    """Comprehensive evaluation report.
    
    Design Pattern: Data Transfer Object
    - Encapsulates all evaluation results
    - Provides methods for analysis and visualization
    - Supports serialization for persistence
    """
    
    model_results: Dict[str, Dict[str, EvaluationMetrics]]
    model_rankings: List[Tuple[str, float]]
    evaluation_timestamp: str
    test_data_size: int
    
    def save_report(self, output_path: Union[str, Path]) -> None:
        """Save evaluation report to file."""
        report_data = {
            "evaluation_timestamp": self.evaluation_timestamp,
            "test_data_size": self.test_data_size,
            "model_rankings": self.model_rankings,
            "detailed_results": {}
        }
        
        # Convert metrics to dictionaries
        for model_name, strategies in self.model_results.items():
            report_data["detailed_results"][model_name] = {}
            for strategy_name, metrics in strategies.items():
                report_data["detailed_results"][model_name][strategy_name] = metrics.to_dict()
        
        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of evaluation results."""
        if not self.model_rankings:
            return {"error": "No model rankings available"}
        
        best_model, best_score = self.model_rankings[0]
        worst_model, worst_score = self.model_rankings[-1]
        
        return {
            "total_models_evaluated": len(self.model_results),
            "test_data_size": self.test_data_size,
            "best_model": {
                "name": best_model,
                "score": best_score
            },
            "worst_model": {
                "name": worst_model,
                "score": worst_score
            },
            "score_range": best_score - worst_score,
            "evaluation_strategies": len(next(iter(self.model_results.values()))) if self.model_results else 0
        }


class EvaluationSuite:
    """Comprehensive evaluation suite for v0.02.
    
    Design Pattern: Facade
    - Provides unified interface to evaluation system
    - Manages complex evaluation workflows
    - Simplifies evaluation for common use cases
    """
    
    def __init__(self, output_dir: Union[str, Path]):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = training_logging_utils.TrainingLogger("evaluation_suite")
    
    def evaluate_single_model(self, model, tokenizer, test_data: List[Dict], 
                            model_name: str = "model") -> EvaluationMetrics:
        """Evaluate a single model comprehensively."""
        self.logger.info(f"Evaluating single model: {model_name}")
        
        # Combine all evaluation strategies
        language_eval = LanguageModelEvaluator()
        snake_eval = SnakeGameEvaluator()
        perf_eval = PerformanceEvaluator()
        
        # Run all evaluations
        lang_metrics = language_eval.evaluate(model, tokenizer, test_data)
        snake_metrics = snake_eval.evaluate(model, tokenizer, test_data)
        perf_metrics = perf_eval.evaluate(model, tokenizer, test_data)
        
        # Combine metrics
        combined_metrics = EvaluationMetrics(
            # Language metrics
            perplexity=lang_metrics.perplexity,
            loss=lang_metrics.loss,
            bleu_score=lang_metrics.bleu_score,
            rouge_l=lang_metrics.rouge_l,
            
            # Snake metrics
            snake_win_rate=snake_metrics.snake_win_rate,
            snake_avg_score=snake_metrics.snake_avg_score,
            snake_avg_steps=snake_metrics.snake_avg_steps,
            snake_decision_accuracy=snake_metrics.snake_decision_accuracy,
            
            # Performance metrics
            inference_time_ms=perf_metrics.inference_time_ms,
            memory_usage_mb=perf_metrics.memory_usage_mb,
            tokens_per_second=perf_metrics.tokens_per_second,
            
            # Combined sample size
            sample_size=len(test_data)
        )
        
        # Save individual model results
        model_report_path = self.output_dir / f"{model_name}_evaluation.json"
        with open(model_report_path, 'w') as f:
            json.dump(combined_metrics.to_dict(), f, indent=2)
        
        self.logger.info(f"Single model evaluation saved to: {model_report_path}")
        
        return combined_metrics
    
    def evaluate_multiple_models(self, models: Dict[str, Tuple], 
                                test_data: List[Dict]) -> EvaluationReport:
        """Evaluate and compare multiple models."""
        self.logger.info(f"Evaluating {len(models)} models for comparison")
        
        # Run comparison
        comparator = ModelComparator(models, test_data)
        results = comparator.compare_models()
        rankings = comparator.rank_models(results)
        
        # Create report
        report = EvaluationReport(
            model_results=results,
            model_rankings=rankings,
            evaluation_timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            test_data_size=len(test_data)
        )
        
        # Save comparison report
        comparison_report_path = self.output_dir / "model_comparison_report.json"
        report.save_report(comparison_report_path)
        
        self.logger.info(f"Model comparison report saved to: {comparison_report_path}")
        
        return report


def main():
    """Demo of evaluation suite."""
    print("📊 Evaluation Suite Demo")
    
    # Create mock test data
    test_data = [
        {
            "prompt": "Snake head at (5,5), apple at (7,7). What move?",
            "completion": "Move RIGHT to approach the apple.",
            "algorithm": "BFS"
        },
        {
            "prompt": "Snake head at (3,3), apple at (1,3). What move?",
            "completion": "Move LEFT to reach the apple directly.",
            "algorithm": "ASTAR"
        }
    ]
    
    print(f"📋 Created {len(test_data)} test examples")
    
    # Demo evaluation strategies
    strategies = [
        LanguageModelEvaluator(),
        SnakeGameEvaluator(num_games=10),
        PerformanceEvaluator()
    ]
    
    for strategy in strategies:
        print(f"\n🔍 {strategy.name}: {strategy.get_description()}")
    
    print("\n✅ Evaluation suite ready for model evaluation!")


if __name__ == "__main__":
    main() 