## VITAL: datasets folder naming/placement

The grid_size should not be fixed to 10, because generated datasets (json files/folders, csv files/folders, jsonl files/folders, etc.) will be stored in ./logs/extensions/datasets/grid-size-N. Make a clear rule and maybe write some python code (maybe OOP Abstract CLass that should be implemented by all extensions blabla-v0.0N? Config constants? Validation mechanism? do it in common folder) for enforcing your rule.


Check this for all heuristics extensions:
- heuristics-v0.01
- heuristics-v0.02
- heuristics-v0.03
- heuristics-v0.04

Check this for all supervized learning extensions:
- supervized-v0.01
- supervized-v0.02
- supervized-v0.03

Check this for all reinforcement learning extensions:
- reinforcement-v0.01
- reinforcement-v0.02

