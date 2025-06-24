Each extension folder of v0.03 (not for v0.01, not for v0.02), will have a folder named "scripts".

For v0.03, it's really important because streamlit app.py will call those scripts extensively. 

### **Direct Script Execution**
```bash
# Train supervised models
python scripts/training/train_supervised.py \
    --dataset-paths logs/extensions/datasets/grid-size-N/blablabla_folder_or_sub_or_subsub_folders_or_file_whose_naming_is_not_decided_yet/*.csv # TODO: check this, update blablabla.
    --model-types all \
    --hyperparameter-tuning \
    --output-dir models/supervised/grid-N/blablabla_folder_or_sub_or_subsub_folders_or_file_whose_naming_is_decided_yet # TODO: check this, update blablabla.

# Generate heuristic datasets
python scripts/data_generation/generate_heuristic_data.py \
    --algorithms bfs astar hamiltonian \
    --grid-size 10 \
    --num-games 1000 \
    --output-format csv

# Evaluate model performance
python scripts/evaluation/evaluate_models.py \
    --model-dir models/supervised/grid-N/blablabla_folder_or_sub_or_subsub_folders_or_file_whose_naming_is_decided_yet # TODO: check this, update blablabla.
    --test-data logs/extensions/datasets/grid-size-N/blablabla_folder_or_sub_or_subsub_folders_or_file_whose_naming_is_decided_yet/test_data.csv \
    --output-dir evaluation_results #TODO: check this, update blablabla.
```



# IMPORTANT
streamlit app.py is not for visualization of game states, is not for real time showing progress, is 
not 
for showing snake moves.

It's main idea is to launch scripts in the folder "scripts" with adjustable params, with 
subprocess. That's why for extensions v0.03 we will have a folder "dashboard" in the first place.
