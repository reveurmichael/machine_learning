## Ducumentation as first priority and first class citizen

Documentation/docstring/comments are very important for me. Each time you do a refactoring, make sure you put back the comments and docstrings. Each time you modify and fix a bug, make sure you record that in related code comments and docstrings. 

## IMPORTANT
We are doing the code refactoring, so as to make things more generic and reusable. Keep the functionality of Task 0 (Task0, LLM playing snake game) unchanged.

## VITAL
Check ROOT/docs/extensions-guideline/project-structure-plan.md. That's our objective of the refactoring.

## Single source of truth
Unless it's between different extensions (each extension, plus the common folder, are regarded as standalone), we should go for single source of truth. Especially from the folder ROOT/config , or from "ROOT/extensions/common/" folder.

## OOP, SOLID and DRY
Respecting OOP, SOLID and DRY principles is very important. Whenever possible, make things OOP, because it is easier to extend. Future tasks such as Task 1, Task 2, Task 3, Task 4, Task 5 (let's call them FUTURE_TASKS) etc. can be implemented as subclasses of base classes (with inheritance, but maybe adaptor/composition as well, though less desirable), or even as subclasses of subclasses of Task 0. We can tolerate that FUTURE_TASKS will not be using all attributes/functions of base classes, as long as this will not pollute the output data files of FUTURE_TASKS.


## Task-0
If you are losing sight of Task-0, check the files:
- logs/hunyuan-t1-latest_20250617_223807/game_1.json
- logs/hunyuan-t1-latest_20250617_223807/game_8.json
- logs/hunyuan-t1-latest_20250617_223807/summary.json
As some of the past experiments logs to help you understand Task-0.
Readme.md is a little bit outdated, but it's a good starting point.

## Class Naming
No need to rename FileManager to Task0FileManager because here by default we are refering to Task-0. So the name GameController, FileManager, GameData, GameLogic, etc. are all Task-0 specific. In extensions we will have Task1-5, which will extend the base classes (BaseFileManager, BaseGameData, BaseGameLogic, etc.).

## Singleton pattern
BaseFileManager and FileManager should use Singleton pattern. Maybe some other classes as well. 

## Design Patterns
As this whole project is about refactoring, we should use design patterns to make things more generic and reusable. Also, this whole project is to be educational, so we should use as many appropriate design patterns as possible to make things more educational. Each time you use a design pattern, you should explain why you are using it, with very detailed comments or docstrings.

## Inheritance
Most likely, classes in "extensions" folder will be inheriting from base classes in "core", "replay" and "gui" folders. In very rare cases, they might be inheriting from the derived classes for Task-0 instead of the base classes.

## No Need for Backward compatibility
We are refactoring with a future proof mindset, to make things look so fresh, so newly shipped, so self-consistent, so self-contained. So we are not going to keep backward compatibility, for anything. Nothing is going to be deprecated, if anything is deprecated, it should be removed. No legacy consideration for extensions. For task0, as long as the output follows the schema of Task-0, we are good. For the schema of Task-0, check the files:
- logs/hunyuan-t1-latest_20250617_223807/game_1.json
- logs/hunyuan-t1-latest_20250617_223807/game_8.json
- logs/hunyuan-t1-latest_20250617_223807/summary.json

## No pollution of code from Task 1-5 to the ROOT (Task-0) folder

No pollution from extensions (Task 1-5) into the ROOT folder (for taks 0, but can be extended to other tasks as well in the folder "extensions"). Hence, such words like "heuristics", "reinforcement learning" should not be used in the ROOT folder. It can only be used in the folder "extensions".

## No unncessary or unimplemented code/functions prepared for Task 1-5 in the ROOT folder but it's not used for Task-0

Let the future tasks (Task 1-5) be the ones to implement the code/functions that they need. Overkill is not good. Over-preparation is not good.

## Class Naming
Regarding the naming: in the root directory, by default it's for task0. so no need for LLMBlabla class  name, just Blabla class name is great. if it's to be extended by task0, 1, 2, 3, 4, 5, then it's name is BaseBlabla. When Extended, it's HeuristicBlabla, RLBlabla, etc.

 

## VITAL: Don't remove any classes in the ./core/, ./replay/ folder. You can add some functions or classes, but don't remove classes.

VITAL: Because it's already being used by extensions (check the folder "./extensions/")


## VITAL: EVOLUTION OF CODE IN DEMONSTRATION: 

Keep the folder "agents" in the folder "./extensions/heuristics-v0.02" and "./extensions/heuristics-v0.03" exactly the same. Though, they each one, as a folder/extension/blabla-v0.0N + the common folder, is standalone.

Keep the folder "agents" in the folder "./extensions/supervized-v0.02" and "./extensions/supervized-v0.03" exactly the same. Though, they each one as a folder/extension/blabla-v0.0N, plus the common folder, is standalone.


Keep the folder "agents" in the folder "./extensions/reinforcement-v0.02" and "./extensions/reinforcement-v0.03" exactly the same. Though, they each one as a folder/extension/blabla-v0.0N, plus the common folder, is standalone.


Keep the folder "agents" in the folder "./extensions/evolutionary-v0.02" and "./extensions/evolutionary-v0.03" exactly the same. Though, they each one as a folder/extension/blabla-v0.0N, plus the common folder, is standalone.


## VITAL: standalone should be very visible, across all extensions, but common folder is important

For somewhat common utils, put things into the ./extensions/common/ folder. We can regard the ./extensions/common/ folder as a folder for somewhat common utils (common for this moment, or maybe will be used in the future), that no one will forget about its presence, then, an extension blabla-v0.0N, plus the common folder, those two together will be regarded as standalone as well. But we should not be sharing code between extensions. It's forbidden. blabla-v0.01 + common is standalone. blabla-v0.02 + common is standalone. blabla-v0.03 + common is standalone. blabla-v0.04 + common is standalone. (though, only heuristics will have v0.04; for other extensions, there is only v0.01, v0.02 and v0.03).

The common folder is important because, after all, each extension blabla-v0.0N, represents important conceptual ideas (e.g. heuristics, supervised learning, RL, etc.), and it's those conceptual ideas that should be highlighed in each extension folder blabla-v0.0N. Moving non-essential code into the common folder helps those conceptual ideas to be more visible.


## Type-hinted

Make code type-hinted, but only where you are really sure of the type. Don't type-hint for the sake of type-hinting.


## IMPORTANT: OOP and docstring, comments
You should never make docstring/comments less clear/verbose/detailed. You should go for OOP and inheritance extensively. You should go for basic but effective Design Patterns extensively and give very good comments/docstrings for each design pattern you use, its motivation, its philosophy, its trade-offs, and why you use it.



## Never should happen
Such code should never happen:
- from heuristics_v0.03 import blabla
- from blablabla_v0.0N import blabla
- from extensions.distillation_v0_03 import blabla
- from extensions.blablabla_v0_0N import blabla


## VITAL

use chdir() extensively, maybe not directly, but you can call functions from "./extensions/common/path_utils.py" and "./utils/path_utils.py"

## DRY principles
For extensions, we should go for DRY principles extensively, as well, but only common utils, and should only be put into the ./extensions/common/ folder. Never share code between extensions. Because, each extension balblabla-v0.0N, plus the common folder, is standalone.

## Extensions v0.02
v0.02 should not break v0.01 functionalities. 

## Extensions v0.03
v0.03 should not break v0.02 functionalities. 

## Extensions v0.04
v0.04 is only for heuristics. For other extensions/algorithms, there is only v0.01, v0.02 and v0.03. Ideally, for heuristics v0.04, it will generate jsonl files, in plus to csv files (of v0.03). You should not break v0.03 functionalities. If you extend thing, use OOP or adapter or create another python file. When finished, try a pipeline to check v0.04 jsonl files are really generated for all those heuristics agents are good. ## No Need for Backward compatibility We are refactoring with a future proof mindset, to make things look so fresh, so needly shipped.. So we are not going to keep backward compatibility, for anything. Nothing is going to be deprecated, if anything is deprecated, it should be removed. No legacy consideration for extensions. You should leave extensive comments/docstrings on those important things to keep in mind. It's like we assume in extensions, blabla-v0.0N we will be able to generate json files, maybe also pth or npz and paquet files in the case of RL/supvervized learning. On the contrary, for transforming those json files into csv, it is using a shared tool in "common" folder. For generating jsonl files, it can be put into the folder of heuristics-v0.04, or maybe the folder "common", depending on which approach gives the best clarity. I find the naming of generate_dataset_v03.py, generate_dataset.py, generate_jsonl_dataset.py python file naming (in the same folder ) really puzzling. For better clarity, put those stuffs into the "common" folder (except, maybe, one things I am not sure, where to put the jsonl generation tool), give really good file naming for python files.


## In extensions folder, if things are to be break, break it.

No need for things like class name aliases, or an adaptpor, etc.

## No import aliases, unless it's really necessary.

Very important.


## VITAL: datasets folder naming/placement, across all extensions

The grid_size should not be fixed to 10, because generated datasets (json files/folders, csv files/folders, jsonl files/folders, etc.) will be stored in ./logs/extensions/datasets/grid-size-N/blabla_v0.0.N_timestamps folder. Make a clear rule and maybe write some python code (maybe OOP Abstract CLass that should be implemented by all extensions blabla-v0.0N? Config constants? Validation mechanism? do it in common folder) for enforcing your rule.

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

## VITAL: models and datasets folder naming/placement, across all extensions

Models trained by machine learning/DL/RL will be stored in ./logs/extensions/models/grid-size-N/blabla_v0.0.N_timestamps folder

Datasets generated by the heuristics/ML/DL will be stored in ./extensions/datasets/grid-size-N/blabla_v0.0.N_timestamps folder



## IMPORTANT
streamlit app.py is not for visualization of game states, is not for real time showing progress, is 
not 
for showing snake moves.

It's main idea is to launch scripts in the folder "scripts" with adjustable params, with 
subprocess. That's why for extensions v0.03 we will have a folder "dashboard" in the first place.

## VITAL
You should never edit the file "ROOT/docs/extensions-guideline/final-decision-1.md" (or, more generally, "ROOT/docs/extensions-guideline/final-decision-N.md"), because it's the final decision, and it's the single source of truth.