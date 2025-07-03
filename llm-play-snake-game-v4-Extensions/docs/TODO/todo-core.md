## 🎯 **核心问题诊断**

当前的架构已经相当出色，但还可以在以下几个方面进一步抽象和封装，让extensions更容易开发：

### 1. **JSON生成管道的抽象化**
目前每个extension都需要实现自己的`generate_game_summary()`方法，存在大量重复代码。

### 2. **统计数据收集的标准化**
各extension需要手动管理step_stats、time_stats等，容易出错。

### 3. **文件管理的统一化**
session summary和game file保存逻辑分散在不同类中。

## 🚀 **建议的重构方案**

### **方案1: 创建通用的JSON Summary生成器**

```python
# 新增：core/game_summary_generator.py
class BaseGameSummaryGenerator:
    """
    通用的游戏摘要生成器，支持所有task类型
    
    设计模式：Template Method Pattern + Strategy Pattern
    目的：标准化JSON生成流程，同时允许task特定的定制
    """
    
    def __init__(self, game_data: "BaseGameData"):
        self.game_data = game_data
        
    def generate_summary(self, **kwargs) -> Dict[str, Any]:
        """生成完整的游戏摘要JSON"""
        summary = {
            # 1. 通用核心字段（所有task共享）
            **self._get_core_fields(),
            
            # 2. task特定字段（子类覆盖）
            **self._get_task_specific_fields(**kwargs),
            
            # 3. 统计数据（自动处理）
            **self._get_statistics_fields(),
            
            # 4. 元数据（标准化）
            **self._get_metadata_fields(**kwargs),
            
            # 5. 回放数据（标准化）
            **self._get_replay_fields(),
        }
        
        # 后处理钩子
        return self._post_process_summary(summary, **kwargs)
    
    def _get_core_fields(self) -> Dict[str, Any]:
        """核心游戏状态字段（所有task通用）"""
        return {
            "score": self.game_data.score,
            "steps": self.game_data.steps,
            "snake_length": self.game_data.snake_length,
            "game_over": self.game_data.game_over,
            "game_end_reason": self.game_data.game_end_reason,
            "round_count": self.game_data.round_manager.round_count,
            "grid_size": getattr(self.game_data, 'grid_size', 10),  # 支持可变网格
        }
    
    def _get_task_specific_fields(self, **kwargs) -> Dict[str, Any]:
        """Task特定字段（子类覆盖）"""
        return {}
    
    def _get_statistics_fields(self) -> Dict[str, Any]:
        """统计数据字段（自动处理）"""
        return {
            "time_stats": self.game_data.stats.time_stats.asdict(),
            "step_stats": self.game_data.stats.step_stats.asdict(),
        }
    
    def _get_metadata_fields(self, **kwargs) -> Dict[str, Any]:
        """元数据字段（标准化）"""
        return {
            "metadata": {
                "timestamp": self.game_data.timestamp,
                "game_number": self.game_data.game_number,
                "round_count": self.game_data.round_manager.round_count,
                **kwargs.get("metadata", {}),
            }
        }
    
    def _get_replay_fields(self) -> Dict[str, Any]:
        """回放数据字段（标准化）"""
        return {
            "detailed_history": {
                "apple_positions": self.game_data.apple_positions,
                "moves": self.game_data.moves,
                "rounds_data": self.game_data.round_manager.get_ordered_rounds_data(),
            }
        }
    
    def _post_process_summary(self, summary: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """后处理钩子（子类可覆盖）"""
        return summary

# Task-0 specific generator
class LLMGameSummaryGenerator(BaseGameSummaryGenerator):
    """LLM专用的摘要生成器"""
    
    def _get_task_specific_fields(self, **kwargs) -> Dict[str, Any]:
        return {
            "llm_info": {
                "primary_provider": kwargs.get("primary_provider"),
                "primary_model": kwargs.get("primary_model"),
                "parser_provider": kwargs.get("parser_provider"),
                "parser_model": kwargs.get("parser_model"),
            },
            "prompt_response_stats": self.game_data.get_prompt_response_stats(),
            "token_stats": self.game_data.get_token_stats(),
        }

# Heuristic specific generator
class HeuristicGameSummaryGenerator(BaseGameSummaryGenerator):
    """启发式算法专用的摘要生成器"""
    
    def _get_task_specific_fields(self, **kwargs) -> Dict[str, Any]:
        return {
            "heuristic_info": {
                "algorithm": getattr(self.game_data, 'algorithm_name', 'BFS'),
                # 移除了LLM相关字段，实现Task-0兼容性
            }
        }
    
    def _post_process_summary(self, summary: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """清理LLM特定字段以实现Task-0兼容性"""
        # 清理rounds_data中的game_state字段
        if "detailed_history" in summary and "rounds_data" in summary["detailed_history"]:
            cleaned_rounds = {}
            for round_key, round_data in summary["detailed_history"]["rounds_data"].items():
                cleaned_round = {
                    "round": round_data.get("round", int(round_key)),
                    "apple_position": round_data.get("apple_position", [0, 0])
                }
                # 确保planned_moves与moves匹配
                if "moves" in round_data:
                    cleaned_round["moves"] = round_data["moves"]
                    cleaned_round["planned_moves"] = round_data["moves"]
                cleaned_rounds[round_key] = cleaned_round
            summary["detailed_history"]["rounds_data"] = cleaned_rounds
        
        return summary
```

### **方案2: 统一的统计数据收集器**

```python
# 新增：core/game_statistics_collector.py
class GameStatisticsCollector:
    """
    统一的统计数据收集器
    
    设计模式：Observer Pattern + Facade Pattern
    目的：自动收集和聚合各类统计数据，减少手动管理
    """
    
    def __init__(self, game_data: "BaseGameData"):
        self.game_data = game_data
        self.collectors = []  # 可插拔的收集器
    
    def add_collector(self, collector: "StatisticsCollector"):
        """添加统计收集器"""
        self.collectors.append(collector)
    
    def record_move(self, move: str, apple_eaten: bool = False):
        """记录移动并自动更新所有统计"""
        # 基础统计
        self.game_data.record_move(move, apple_eaten)
        
        # 通知所有收集器
        for collector in self.collectors:
            collector.on_move(move, apple_eaten, self.game_data)
    
    def record_game_end(self, reason: str):
        """记录游戏结束并聚合统计"""
        self.game_data.record_game_end(reason)
        
        # 通知所有收集器
        for collector in self.collectors:
            collector.on_game_end(reason, self.game_data)

class StatisticsCollector(ABC):
    """统计收集器接口"""
    
    @abstractmethod
    def on_move(self, move: str, apple_eaten: bool, game_data: "BaseGameData"):
        pass
    
    @abstractmethod
    def on_game_end(self, reason: str, game_data: "BaseGameData"):
        pass

class HeuristicStatisticsCollector(StatisticsCollector):
    """启发式算法专用统计收集器"""
    
    def on_move(self, move: str, apple_eaten: bool, game_data: "BaseGameData"):
        # 自动分类移动类型
        if move == "INVALID_REVERSAL":
            game_data.stats.step_stats.invalid_reversals += 1
        elif move == "NO_PATH_FOUND":
            game_data.stats.step_stats.no_path_found += 1
        elif move in ["UP", "DOWN", "LEFT", "RIGHT"]:
            game_data.stats.step_stats.valid += 1
    
    def on_game_end(self, reason: str, game_data: "BaseGameData"):
        # 启发式特定的游戏结束处理
        if hasattr(game_data, 'algorithm_name'):
            print_info(f"[{game_data.algorithm_name}] Game ended: {reason}")
```

### **方案3: 统一的文件管理器**

```python
# 增强：core/game_file_manager.py 
class UniversalFileManager(BaseFileManager):
    """
    通用文件管理器，支持所有task类型
    
    设计模式：Factory Method Pattern + Template Method Pattern
    目的：统一文件操作接口，自动选择合适的处理器
    """
    
    def save_game_files(self, game_data: "BaseGameData", log_dir: str, **kwargs):
        """统一保存游戏文件（game_N.json + session更新）"""
        # 1. 生成适当的摘要生成器
        generator = self._create_summary_generator(game_data)
        
        # 2. 生成game_N.json
        game_file = self.get_game_json_filename(game_data.game_number)
        game_path = os.path.join(log_dir, game_file)
        
        summary = generator.generate_summary(**kwargs)
        
        # 3. 保存文件
        os.makedirs(os.path.dirname(game_path), exist_ok=True)
        with open(game_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, cls=NumPyJSONEncoder, indent=2)
        
        # 4. 更新session summary
        self._update_session_summary(log_dir, game_data, **kwargs)
        
        return summary
    
    def _create_summary_generator(self, game_data: "BaseGameData") -> BaseGameSummaryGenerator:
        """工厂方法：根据game_data类型创建适当的生成器"""
        if isinstance(game_data, GameData):  # Task-0
            return LLMGameSummaryGenerator(game_data)
        elif hasattr(game_data, 'algorithm_name'):  # Heuristics
            return HeuristicGameSummaryGenerator(game_data)
        else:
            return BaseGameSummaryGenerator(game_data)  # 通用
    
    def _update_session_summary(self, log_dir: str, game_data: "BaseGameData", **kwargs):
        """更新session级别的summary.json"""
        # 根据task类型选择适当的stats manager
        if isinstance(game_data, GameData):
            stats_manager = GameStatsManager()
        else:
            stats_manager = BaseGameStatsManager()
        
        # 自动聚合统计数据
        session_stats = self._collect_session_stats(game_data, **kwargs)
        stats_manager.save_session_stats(log_dir, **session_stats)
```

### **方案4: 增强BaseGameData的抽象层**

```python
# 增强：core/game_data.py BaseGameData类
class BaseGameData:
    def __init__(self) -> None:
        self.reset()
        
        # 新增：统一的文件管理器
        self._file_manager = UniversalFileManager()
        
        # 新增：统计收集器
        self._stats_collector = GameStatisticsCollector(self)
        self._register_default_collectors()
    
    def _register_default_collectors(self):
        """注册默认的统计收集器（子类可覆盖）"""
        pass
    
    def save_game_summary(self, filepath: str, **kwargs) -> Dict[str, Any]:
        """统一的游戏摘要保存方法"""
        log_dir = os.path.dirname(filepath)
        return self._file_manager.save_game_files(self, log_dir, **kwargs)
    
    def record_move(self, move: str, apple_eaten: bool = False) -> None:
        """增强的移动记录方法"""
        # 使用统一的统计收集器
        self._stats_collector.record_move(move, apple_eaten)
    
    def record_game_end(self, reason: str) -> None:
        """增强的游戏结束记录方法"""
        self._stats_collector.record_game_end(reason)

# Task特定的实现
class HeuristicGameData(BaseGameData):
    def _register_default_collectors(self):
        """注册启发式算法特定的收集器"""
        self._stats_collector.add_collector(HeuristicStatisticsCollector())
        
    def _create_summary_generator(self) -> BaseGameSummaryGenerator:
        """覆盖以使用启发式专用生成器"""
        return HeuristicGameSummaryGenerator(self)
```

## 🎯 **实施优先级建议**

### **Phase 1: 立即实施（最大收益）**
1. **创建BaseGameSummaryGenerator**：消除JSON生成的重复代码
2. **增强BaseGameData的save方法**：统一文件保存接口

### **Phase 2: 中期实施**
3. **统一统计收集器**：标准化统计数据管理
4. **增强UniversalFileManager**：统一文件操作

### **Phase 3: 长期优化**
5. **添加扩展点和钩子**：为未来task类型提供更好支持

## 📊 **预期收益**

### **对Extensions的好处：**
- ✅ **减少90%的JSON生成代码**：只需配置字段映射
- ✅ **自动统计管理**：无需手动追踪step_stats等
- ✅ **统一文件API**：一个方法搞定所有文件操作
- ✅ **Task-0完全兼容**：自动处理兼容性问题

### **对Task-0的好处：**
- ✅ **零功能影响**：现有代码继续正常工作
- ✅ **更好的代码组织**：逻辑更清晰，更易维护
- ✅ **更强的扩展性**：为未来功能提供更好基础

这种重构方案遵循了SOLID原则和现有的设计模式，同时为extensions提供了更简洁、更强大的开发体验。