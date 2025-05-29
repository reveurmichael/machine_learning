In the results_for_comparison folder, we have the results of the experiments. do a check there for the filenames. what we want: for each experiment, for each folder of "prompts" and "responses", we want to rename the files, so that for each N (1, 2, 3, ..), we will have gameN_roundX_blabla.txt, with, X going from 1, 2, 3, ... So here how to get X to change is the key, as N is already there.Note that we are not here changing the source code of the project. We are just adding a post-processing script in results_for_comparison/rename_files.py. I run the script, and the txt file names are changed. Do it for me.



This time, things are different. We will need to change the 


Then, change the code so that round_X will be great. 
