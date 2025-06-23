## general

v0.03 will be able to generate csv dataset, with no language sentences.


v0.03 will also feature streamlit app.py, with a tab for each algorithm. 

## heuristics 

In v0.03, in the folder "./extensions/heuristics-v0.03", we can extend things in v0.02. The point is to show that from v0.02, there is progression, that's how softwares/code/systems evolve.

What is added:
- app.py , a streamlit app for lauching those different agents (with one or several different arguments, at least the max-games argument), and shows replay (with pygame and flask web mode, both). You can check app.py in the ROOT for task0 for some inspiration (check the folder "dashboard" in the ROOT, the session_utils.py in the folder "utils" in the ROOT).
- We have to add things like replay mode code. But we will really re-use very very extensively what is provided in those base classes and utils from Task0 in ROOT directory. 


Do it now, in "./extensions/heuristics-v0.03". I have already copied the code from "./extensions/heuristics-v0.02" to "./extensions/heuristics-v0.03". You can and should improve the code in "./extensions/heuristics-v0.03".

We will also put all those agents in the folder "./extensions/heuristics-v0.03/agents" .
I have also moved main.py into scripts folder. app.py will be the entry point for users.


you should notice this word I mentioned: web, flask. so you will have scripts/replay_web.py and scirpts/replay.py, just like for TASK0 . FOr web, you can go mvc, or simpler, it's up to you.

v0.02 and v0.03 should have exactly the same code in their respective agents folders.

v0.03 will also be able to generate csv dataset, with no language sentences.

The csv dataset should be in logs/extensions/dataset/grid-size-N  where N is the size of the grid (10 for this moment by default, but can be configurable in heuristics/supversized learning main.py main_web.py scripts).


## supervized learning models

supvervized learning models (let's say, xgboost, lightgbm, neural networks (pytorch version), cnn (pytorch version), rnn (pytorch version), GNN (pytorch version, pytorch geometric), etc):
- Just like for heuristics, we will have v0.01, v0.02, v0.03, etc.
- For this moment, go for only v0.01 and v0.02.
- v0.01 is for neural networks (pytorch version)
- v0.02 is for all supervised learning models, like xgboost, lightgbm,neural network, cnn, rnn, GNN, etc. but without the gui, no replay, no streamlit app.py yet.
- v0.03 is will have gui, replay, web mode, pygame mode, streamlit app.py.

In the app.py, we distingush between 
- training 
- using the models to generate game_N.json and summary.json as well as csv dataset.
