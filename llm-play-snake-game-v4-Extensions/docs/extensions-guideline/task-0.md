Task-0 functionality should never be changed: All Task-0 (LLM Snake Game, located in ROOT folder, excluding the ROOT/extensions folder) functionality should remain intact. Existing Task-0 scripts and workflows should continue to work. No touch to Task-0 code.

Base Class Pattern: Task-0 classes should inherit from base classes without functionality loss

Output Schema: Game logs should still follow the original JSON schema format, as in:
- logs/hunyuan-t1-latest_20250617_223807/game_1.json
- logs/hunyuan-t1-latest_20250617_223807/game_8.json
- logs/hunyuan-t1-latest_20250617_223807/summary.json

For task0, as long as the output follows the schema of Task-0, we are good. For the schema of Task-0, check the files:
- logs/hunyuan-t1-latest_20250617_223807/game_1.json
- logs/hunyuan-t1-latest_20250617_223807/game_8.json
- logs/hunyuan-t1-latest_20250617_223807/summary.json