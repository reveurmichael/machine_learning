
check the logs in   , its game_N.json and summary.json and the console output to see if everything is great. What else code should we modify? just answer the question, no need to modify the code.



for web mode, since we have already human_play_web.py and replay_web.py, should we have main_web.py? will it be a good idea? will it be a lot of code changes?

without changing any functionality in human_play_web.py, replay_web.py, main.py mode, go ahead write the code for main_web.py. main_web.py should have the same argparse options as main.py (plus host/port.). You can get a lot of inspiration from replay_web.py and human_play_web.py. No need for In-browser control panel, no need for "If you want “continue” and “replay” links after the run, add a small REST route.". UI similar to human_play_web.py/replay_web.py and reuse the same code for the web mode as much as possible. 



Why do we have game_manager.parser_usage_count in continuation_utils.py ? everything related to parser usage should be removed. 