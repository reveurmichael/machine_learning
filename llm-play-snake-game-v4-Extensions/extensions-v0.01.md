SO here is what I want to do for a first extension, for this very first version, intentionally simple, extremely simple. 

v0.01 will be called "heuristics-v0.01". It will use the API (extending Base Classes, in the core folder) in the current root directory, for sure. But it will be a very simple heuristic (let's say BFS or DFS). It will not be using the pygame or the web UI. if possible, there is even no need need use the --no-gui flag or use_gui flag, because it will be no gui by default.


It will generating game_N.json files and summary.json files. But, per the implementation of heuristics-v0.01, there is no Replay mode yet, no web mode yet. no pygame or web mode yet. just a script of heuristic BFS or DFS for generating those game_N.json and summary.json files. It will of course extend the BaseController class, maybe other classes as well. 


the folder will be named "./extensions/heuristics-v0.01" . 
python code will be within the folder.



Good. Now, for ./extensions/heuristics-v0.02 (A star) and ./extensions/heuristics-v0.03 (hamilton cycle), we will do the best heuristics that we know for playing snake games, with the same implementation structure as ./extensions/heuristics-v0.01. No need to create the ./extensions/common/ folder because I want things to be standalone, each and every of them. 






