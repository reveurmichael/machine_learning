## heuristics 

v0.01 has been done and is called "./extensions/heuristics-v0.01". I want to show that after this first proof of concept done in v0.01, we can extend things in v0.02, in the folder "./extensions/heuristics-v0.02", mainly by adding several other heuristic agents (in the folder ./extensions/heuristics-v0.02/agents). Here is the list of heuristic agents in v0.02:
- agent_bfs.py the same as in v0.01, pure bfs, existing code GOOD
- agent_bfs_safe_greedy.py : an improved version of agent_bfs.py . , existing code GOOD except maybe the naming of classes or attributs.
- agent_bfs_hamiltonian.py : an even more improved version of agent_bfs_safe_greedy.py, by introducing hamiltonian into bfs and safe_greedy. existing code might be GOOD. Improve the naming of classes or attributs
- agent_dfs.py : a version of dfs.py (pure dfs, no code yet)
- agent_astar.py : a version of astar.py (pure astar, existing code might be good, at least it runs well)
- agent_astar_hamiltonian.py : an improved version of agent_astar.py, by introducing hamiltonian into astar. Existing code might be OK.
- agent_hamiltonian.py : a version of hamiltonian.py (pure hamiltonian, no code yet but you can get some inspiration from agent_astar_hamiltonian.py , maybe.)


Since agent_bfs_safe_greedy.py is an upgrade on agent_bfs.py, it will either extend agent_bfs.py, or be much longer than agent_bfs.py.
The same goes for: 
- agent_bfs.py and agent_bfs_hamiltonian.py
- agent_astar.py and agent_astar_hamiltonian.py
- agent_hamiltonian.py and agent_astar_hamiltonian.py
- agent_hamiltonian.py and agent_bfs_hamiltonian.py

after the modification, just copy the agents folder from v0.02's agents folder to v0.03's agents folder, or the inverse: copy the agents folder from v0.03's agents folder to v0.02's agents folder.

we will have "--algorithm" argument, maybe "--verbose" argument, maybe a little bit of other argment. The point is to show that from v0.01, there is progression, that's how softwares/code/systems evolve.

There is no need for :
 self.game.game_state.save_game_summary(
            game_filepath,
            primary_provider=None,
            primary_model=None,
            parser_provider=None,
            parser_model=None,
        )
No need for replay-compatible with Task-0 . Remove those code, those entries in json files.

It will generating game_N.json files and summary.json files. But, per the implementation of heuristics-v0.02, there is no Replay mode yet, no web mode yet. no pygame or web mode yet. 

the folder will be named "./extensions/heuristics-v0.02" . python code will be within the folder. Some code exists already, but we will improve them.


after the modification, just copy the agents folder from v0.02 to v0.03. 

## supervized learning models







