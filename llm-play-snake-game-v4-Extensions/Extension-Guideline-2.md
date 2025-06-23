
I am now thinking about using the heuristics-v0.03 extension to generate data, for training supvervized learning models (let's say, xgboost, lightgbm, neural networks (pytorch version), cnn (pytorch version), rnn (pytorch version), GNN (pytorch version, pytorch geometric), etc).
- question: how should the dataset be stored? csv? other stuff? let's call it DATA_FORMAT.
- question: how should the dataset be structured/encoded? let's call it DATA_STRUCTURE.
- This game_N.json to DATA_FORMAT.DATA_STRUCTURE function, called in v0.03 (not v0.01, not v0.02), will be used by other future extensions as well. So it's good to have it in the common folder.
- supervized learning algorithm, when go for training, can use option --dataset-path to load the dataset (in the ./ROOT/logs/extensions/ folder), with variable length (e.g. --dataset-path dir1 dir2 dir3 ... )


supvervized learning models (let's say, xgboost, lightgbm, neural networks (pytorch version), cnn (pytorch version), rnn (pytorch version), GNN (pytorch version, pytorch geometric), etc):
- Just like for heuristics, we will have v0.01, v0.02, v0.03, etc.
- For this moment, go for only v0.01 and v0.02.
- v0.01 is for neural networks (pytorch version)
- v0.02 is for all supervised learning models, like xgboost, lightgbm,neural network, cnn, rnn, GNN, etc. but without the gui, no replay, no streamlit app.py yet.
- v0.03 is will have gui, replay, web mode, pygame mode, streamlit app.py.

The other thing I am thinking about, for heurstics v0.04, which is based on v0.03, hence is, since I will be having Task4: fine tuning LLM models, I will have to have long long long language sentences telling how the heuristics working (inserted into each key decision point of the heuristics, so the when fine tuning LLM models, we can have the language sentences telling how the heuristics working), guiding LLM's to learn how to play snake game.




We will have GENETIC ALGORITHM/EVOLUTIONARY ALGORITHM. will have v0.01, v0.02, v0.03, as well. But this one won't be able to generate language sentences. They will be similar to heuristics v0.01, v0.02, v0.03.

