"""
Replay module for the Snake game.
Allows replaying of previously recorded games based on logged moves.
"""

import os
import re
import json
import time
import argparse
import pygame
import numpy as np
import glob
from pathlib import Path
from snake_game import SnakeGame
from config import MOVE_PAUSE, DIRECTIONS
from gui import DrawWindow

def extract_moves_from_log_dir(log_dir, game_number=None):
    """Extract the sequence of moves from log files in a directory.
    
    Args:
        log_dir: Path to the log directory
        game_number: Specific game number to extract moves for (None means all games)
        
    Returns:
        Dictionary mapping game numbers to move lists, or a single list of moves if game_number is specified
    """
    all_game_moves = {}
    log_dir_path = Path(log_dir)
    
    # Check if responses directory exists in the log directory
    responses_dir = log_dir_path / "responses"
    if responses_dir.exists() and responses_dir.is_dir():
        response_pattern = str(responses_dir / "response_*.txt")
    else:
        # Fall back to the main directory if responses subdirectory doesn't exist
        response_pattern = str(log_dir_path / "response_*.txt")
    
    # Get all response files
    response_files = sorted(glob.glob(response_pattern))
    
    if not response_files:
        print(f"No response files found in {log_dir}")
        return {} if game_number is None else []
    
    # Process each response file
    for response_file in response_files:
        # Extract game number from filename
        file_game_num_match = re.search(r'response_(\d+)\.txt', os.path.basename(response_file))
        if not file_game_num_match:
            continue
            
        file_game_num = int(file_game_num_match.group(1))
        
        # If a specific game number is requested, skip other games
        if game_number is not None and file_game_num != game_number:
            continue
        
        # Extract moves from this file
        moves = []
        try:
            with open(response_file, 'r', encoding='utf-8') as f:
                log_content = f.read()
            
            # Try to find JSON responses in the log
            json_pattern = r'\{[\s\S]*?"moves"\s*:\s*\[([\s\S]*?)\][\s\S]*?\}'
            json_matches = re.finditer(json_pattern, log_content)
            
            for match in json_matches:
                json_str = match.group(0)
                try:
                    # Try to parse the JSON
                    json_data = json.loads(json_str)
                    if "moves" in json_data and isinstance(json_data["moves"], list):
                        # Extract the moves
                        valid_moves = [move.upper() for move in json_data["moves"] 
                                      if isinstance(move, str) and move.upper() in ["UP", "DOWN", "LEFT", "RIGHT"]]
                        if valid_moves:
                            moves.extend(valid_moves)
                except json.JSONDecodeError:
                    # Try to parse a more specific part of the match
                    moves_array = match.group(1)
                    # Extract all quoted strings (both single and double quotes)
                    move_matches = re.findall(r'["\']([^"\']+)["\']', moves_array)
                    valid_moves = [move.upper() for move in move_matches 
                                  if move.upper() in ["UP", "DOWN", "LEFT", "RIGHT"]]
                    if valid_moves:
                        moves.extend(valid_moves)
                
            if moves:
                # Store the moves for this game
                all_game_moves[file_game_num] = moves
                print(f"Extracted {len(moves)} moves from game {file_game_num}")
            
        except Exception as e:
            print(f"Error extracting moves from {response_file}: {e}")
    
    # Check for game summary file
    if game_number is not None:
        json_summary_file = log_dir_path / f"game{game_number}_summary.json"
        if json_summary_file.exists():
            try:
                with open(json_summary_file, 'r', encoding='utf-8') as f:
                    summary_data = json.load(f)
                print(f"Found JSON summary file for game {game_number}")
                # Could extract additional information from JSON summary if needed
            except Exception as e:
                print(f"Error reading JSON summary file: {e}")
    
    # Return the appropriate result based on the input
    if game_number is not None:
        return all_game_moves.get(game_number, [])
    else:
        return all_game_moves

def extract_apple_positions(log_dir, game_number):
    """Extract apple positions from a game summary file.
    
    Args:
        log_dir: Path to the log directory
        game_number: Game number to extract apple positions for
        
    Returns:
        List of apple positions as [x, y] arrays
    """
    log_dir_path = Path(log_dir)
    json_summary_file = log_dir_path / f"game{game_number}_summary.json"
    apple_positions = []
    
    if not json_summary_file.exists():
        print(f"No JSON summary file found for game {game_number}")
        return apple_positions
    
    try:
        with open(json_summary_file, 'r', encoding='utf-8') as f:
            summary_data = json.load(f)
        
        # Extract apple positions from JSON
        if 'apple_positions' in summary_data and summary_data['apple_positions']:
            # Convert string positions like "(x,y)" to arrays [x,y]
            for pos_str in summary_data['apple_positions']:
                match = re.match(r'\((\d+),(\d+)\)', pos_str)
                if match:
                    x, y = int(match.group(1)), int(match.group(2))
                    apple_positions.append(np.array([x, y]))
        
        print(f"Extracted {len(apple_positions)} apple positions from game {game_number} JSON summary")
    
    except Exception as e:
        print(f"Error extracting apple positions from JSON summary: {e}")
    
    return apple_positions

def filter_invalid_moves(moves):
    """Filter out invalid moves (e.g., reversals) from the sequence.
    
    Args:
        moves: List of moves
        
    Returns:
        Filtered list of moves
    """
    if not moves:
        return []
        
    filtered_moves = [moves[0]]
    
    for i in range(1, len(moves)):
        current = moves[i]
        previous = filtered_moves[-1]
        
        # Skip invalid reversals
        if (previous == "UP" and current == "DOWN") or \
           (previous == "DOWN" and current == "UP") or \
           (previous == "LEFT" and current == "RIGHT") or \
           (previous == "RIGHT" and current == "LEFT"):
            print(f"Skipping invalid reversal: {previous} -> {current}")
            continue
            
        filtered_moves.append(current)
    
    return filtered_moves

def replay_game(moves, apple_positions=None, move_pause=MOVE_PAUSE, game_number=None):
    """Replay a game using the provided sequence of moves and apple positions.
    
    Args:
        moves: List of moves to replay
        apple_positions: List of apple positions to use in replay
        move_pause: Pause between moves in seconds
        game_number: Game number being replayed (for display purposes)
        
    Returns:
        Final score of the game
    """
    # Initialize pygame
    pygame.init()
    pygame.font.init()
    
    game_title = "Snake Game - Replay Mode"
    if game_number is not None:
        game_title += f" (Game {game_number})"
    pygame.display.set_caption(game_title)
    
    # Initialize game
    game = SnakeGame()
    
    # Enable replay mode and set apple positions if provided
    if apple_positions and len(apple_positions) > 0:
        game.set_replay_mode(True)
        # We need to set the initial apple position and populate the history
        game.apple_positions_history = []
        for pos in apple_positions:
            game.apple_positions_history.append(pos)
        
        # Set the first apple position
        if len(apple_positions) > 0:
            game.set_apple_position(apple_positions[0])
            print(f"Set initial apple position to ({apple_positions[0][0]}, {apple_positions[0][1]})")
    
    # Set up game variables
    clock = pygame.time.Clock()
    running = True
    game_active = True
    
    # Process moves to filter out invalid ones
    processed_moves = filter_invalid_moves(moves)
    move_index = 0
    
    # Display information
    print(f"Replaying game with {len(processed_moves)} moves")
    if game_number is not None:
        print(f"Game number: {game_number}")
    
    # Main game loop
    while running and move_index < len(processed_moves):
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    # Pause/resume on space
                    time.sleep(1)
        
        if game_active:
            # Get the next move
            next_move = processed_moves[move_index]
            print(f"Move {move_index+1}/{len(processed_moves)}: {next_move}")
            
            # Execute the move
            game_active, apple_eaten = game.make_move(next_move)
            
            # Update the game state
            game.update()
            
            # Set custom text for replay mode
            replay_info = f"REPLAY MODE - Game {game_number}" if game_number else "REPLAY MODE"
            move_info = f"Move: {move_index+1}/{len(processed_moves)}"
            custom_text = f"{replay_info}\n{move_info}"
            
            # Use the existing draw method, adding custom text if possible
            # This minimizes code changes by using the existing draw interface
            game.processed_response = custom_text
            game.draw()
            
            # Increment move index
            move_index += 1
            
            # Pause between moves
            time.sleep(move_pause)
        
        if not game_active:
            print(f"Game over! Score: {game.score}")
            time.sleep(2)  # Pause before exiting
            break
    
    # Clean up
    pygame.quit()
    
    return game.score

def replay_all_games(log_dir, move_pause=MOVE_PAUSE):
    """Replay all games from a log directory.
    
    Args:
        log_dir: Path to the log directory
        move_pause: Pause between moves in seconds
        
    Returns:
        Dictionary mapping game numbers to scores
    """
    all_game_moves = extract_moves_from_log_dir(log_dir)
    
    if not all_game_moves:
        print("No games found to replay")
        return {}
    
    scores = {}
    for game_number, moves in sorted(all_game_moves.items()):
        if moves:
            print(f"\n===== Replaying Game {game_number} =====")
            
            # Extract apple positions for this game
            apple_positions = extract_apple_positions(log_dir, game_number)
            
            # Replay the game with apple positions
            score = replay_game(moves, apple_positions, move_pause, game_number)
            scores[game_number] = score
            print(f"Game {game_number} complete. Score: {score}")
    
    return scores

def parse_arguments():
    """Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Replay a Snake game from log files')
    parser.add_argument('--log-dir', type=str, required=True,
                      help='Path to the log directory containing game responses')
    parser.add_argument('--game', type=int, default=None,
                      help='Specific game number to replay (default: replay all games)')
    parser.add_argument('--move-pause', type=float, default=MOVE_PAUSE,
                      help=f'Pause between moves in seconds (default: {MOVE_PAUSE})')
    
    return parser.parse_args()

def main():
    """Main function for the replay script."""
    args = parse_arguments()
    
    if args.game is not None:
        # Replay a specific game
        print(f"Replaying game {args.game} from {args.log_dir}")
        moves = extract_moves_from_log_dir(args.log_dir, args.game)
        
        if not moves:
            print(f"No valid moves found for game {args.game}")
            return
        
        # Extract apple positions for this game
        apple_positions = extract_apple_positions(args.log_dir, args.game)
        
        # Replay the game
        final_score = replay_game(moves, apple_positions, args.move_pause, args.game)
        print(f"Replay of game {args.game} complete. Final score: {final_score}")
    else:
        # Replay all games
        print(f"Replaying all games from {args.log_dir}")
        scores = replay_all_games(args.log_dir, args.move_pause)
        
        if scores:
            print("\n===== Replay Summary =====")
            for game_number, score in sorted(scores.items()):
                print(f"Game {game_number}: Score {score}")
        else:
            print("No games were replayed")

if __name__ == "__main__":
    main()
