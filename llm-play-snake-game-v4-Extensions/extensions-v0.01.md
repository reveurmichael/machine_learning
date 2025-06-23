## heuristics 

SO here is what I want to do for a first extension, for this very first version, intentionally simple, extremely simple. 

v0.01 will be called "./extensions/heuristics-v0.01". It will use the API (extending Base Classes, in the core folder) in the current root directory, for sure. But it will be a very simple heuristic (let's say BFS). It will not be using the pygame or the web UI. if possible, there is even no need need use the --no-gui flag or use_gui flag, because it will be no gui by default. We know we have the code already, but I want things to be even more simpler. Specifically for v0.01, there is even no "--algorithm" argument, no "--log-dir", no "--verbose" argument,  for main.py. Specifically for v0.01, there is even no 

there is only one agent in v0.01, namely agent_bfs.py . Attention, the python file name is agent_bfs.py, not bfs_agent.py. 

There is no need for :
 self.game.game_state.save_game_summary(
            game_filepath,
            primary_provider=None,
            primary_model=None,
            parser_provider=None,
            parser_model=None,
        )
No need for replay-compatible with Task-0 . Remove those code, those entries in json files.

In a lot of sense, you can and should make the code in  "./extensions/heuristics-v0.01" even simpler, much simpler, because it's just a proof of concept that things can go well with our abstraction prepared by Task0 in the ROOT directory.

It will generating game_N.json files and summary.json files. But, per the implementation of heuristics-v0.01, there is no Replay mode yet, no web mode yet. no pygame or web mode yet. just a script of heuristic BFS for generating those game_N.json and summary.json files. It will of course extend the BaseGameController or BaseGameLogic class, maybe other a little classes as well (as small number as possible). 

the folder will be named "./extensions/heuristics-v0.01" . python code will be within the folder.


## supervized learning models






