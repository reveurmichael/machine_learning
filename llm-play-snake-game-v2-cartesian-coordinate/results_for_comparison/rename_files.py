"""
Script to rename files in the results_for_comparison directory.
Renames files to have sequential round numbers for each game.
"""

import os
import re
import glob
from collections import defaultdict

def rename_files_in_directory(directory):
    """
    Renames files in the given directory to have sequential round numbers.
    
    Args:
        directory: Directory containing files to rename
    """
    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        return 0
    
    # Get all text files in the directory
    files = glob.glob(os.path.join(directory, "*.txt"))
    
    # Group files by game number
    game_files = defaultdict(list)
    
    # Regular expression to extract game number and round number
    pattern = re.compile(r'game(\d+)_round(\d+)_(\w+)\.txt')
    
    for file_path in files:
        filename = os.path.basename(file_path)
        match = pattern.match(filename)
        if match:
            game_num = int(match.group(1))
            round_num = int(match.group(2))
            file_type = match.group(3)  # prompt or response
            game_files[game_num].append((round_num, file_type, file_path))
    
    # Rename files for each game
    renamed_count = 0
    for game_num, files_info in game_files.items():
        # Sort files by original round number to maintain order
        files_info.sort(key=lambda x: x[0])
        
        # Rename each file with sequential round numbers
        for new_round_num, (old_round_num, file_type, file_path) in enumerate(files_info, 1):
            if old_round_num == new_round_num:
                continue  # Skip if the round number is already correct
                
            # Create new filename with sequential round number
            filename = os.path.basename(file_path)
            new_filename = f"game{game_num}_round{new_round_num}_{file_type}.txt"
            new_file_path = os.path.join(os.path.dirname(file_path), new_filename)
            
            # Rename the file
            os.rename(file_path, new_file_path)
            renamed_count += 1
            print(f"Renamed: {filename} -> {new_filename}")
    
    return renamed_count

def main():
    """Main function to rename files in all experiment directories."""
    base_dir = "results_for_comparison"
    if not os.path.exists(base_dir):
        print(f"Base directory not found: {base_dir}")
        return
    
    # Get all experiment directories
    experiment_dirs = [d for d in os.listdir(base_dir) 
                      if os.path.isdir(os.path.join(base_dir, d))]
    
    total_renamed = 0
    
    for exp_dir in experiment_dirs:
        exp_path = os.path.join(base_dir, exp_dir)
        print(f"\nProcessing experiment directory: {exp_dir}")
        
        # Process prompts directory
        prompts_dir = os.path.join(exp_path, "prompts")
        if os.path.exists(prompts_dir):
            renamed = rename_files_in_directory(prompts_dir)
            if renamed > 0:
                print(f"Renamed {renamed} files in prompts directory")
            total_renamed += renamed
        
        # Process responses directory
        responses_dir = os.path.join(exp_path, "responses")
        if os.path.exists(responses_dir):
            renamed = rename_files_in_directory(responses_dir)
            if renamed > 0:
                print(f"Renamed {renamed} files in responses directory")
            total_renamed += renamed
    
    print(f"\nTotal files renamed: {total_renamed}")

if __name__ == "__main__":
    main() 