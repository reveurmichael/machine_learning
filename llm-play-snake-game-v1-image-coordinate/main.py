"""
Main entry point for the LLM-controlled Snake game.
This script initializes the game, sets up the LLM client, and runs the main game loop.
"""

import time
import pygame
import sys
import argparse
import os
from datetime import datetime
from snake_game import SnakeGame
from llm_client import LLMClient
from config import TIME_DELAY, TIME_TICK, PROMPT_TEMPLATE
from colorama import Fore, init as init_colorama

# Initialize colorama for colored terminal output
init_colorama(autoreset=True)

# Pause time between sequential moves (in seconds)
MOVE_PAUSE = 1.0

def save_to_file(content, directory, filename):
    """Save content to a file.
    
    Args:
        content: Content to save
        directory: Directory to save to
        filename: Name of the file
    """
    os.makedirs(directory, exist_ok=True)
    file_path = os.path.join(directory, filename)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    return file_path

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='LLM-guided Snake game')
    parser.add_argument('--provider', type=str, default='hunyuan',
                      help='LLM provider to use (hunyuan, ollama, deepseek, or mistral)')
    parser.add_argument('--model', type=str, default=None,
                      help='Specific model to use with the provider (e.g., llama3.2:latest for Ollama, deepseek-chat for Deepseek, mistral-medium-latest for Mistral)')
    parser.add_argument('--max-games', type=int, default=6,
                      help='Maximum number of games to play')
    parser.add_argument('--move-pause', type=float, default=MOVE_PAUSE,
                      help='Pause between sequential moves in seconds (default: 1.0)')
    
    return parser.parse_args()

def main():
    """Initialize and run the LLM Snake game."""
    # Parse command line arguments
    args = parse_arguments()
    
    try:
        # Initialize pygame
        pygame.init()
        pygame.font.init()
        
        # Set up the game
        game = SnakeGame()
        
        # Set up the LLM client
        llm_client = LLMClient(provider=args.provider)
        print(Fore.GREEN + f"✅ Using LLM provider: {args.provider}")
        if args.model:
            print(Fore.GREEN + f"🤖 Using model: {args.model}")
        print(Fore.GREEN + f"⏱️ Pause between moves: {args.move_pause} seconds")
        
        # Set up logging directories
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = f"{timestamp}"
        prompts_dir = os.path.join(log_dir, "prompts")
        responses_dir = os.path.join(log_dir, "responses")
        
        # Game variables
        speed_up = False
        time_delay = TIME_DELAY
        time_tick = TIME_TICK
        clock = pygame.time.Clock()
        game_count = 0
        game_active = True
        round_count = 0
        game_round_count = {}  # Dictionary to store round count per game
        need_new_plan = True
        
        # Main game loop
        running = True
        while running and game_count < args.max_games:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        # Toggle speed
                        speed_up = not speed_up
                        if speed_up:
                            print(Fore.YELLOW + "⚡ Speed mode enabled")
                            time_delay, time_tick = 0, 0
                        else:
                            print(Fore.BLUE + "🐢 Normal speed mode")
                            time_delay, time_tick = TIME_DELAY, TIME_TICK
                    elif event.key == pygame.K_r:
                        # Reset game
                        game.reset()
                        game_active = True
                        need_new_plan = True
                        print(Fore.GREEN + "🔄 Game reset")
            
            if game_active:
                try:
                    # Check if we need to request a new plan from the LLM
                    if need_new_plan:
                        # Get game state
                        game_state = game.get_state_representation()
                        
                        # Format prompt for LLM
                        prompt = PROMPT_TEMPLATE.format(
                            game_state=game_state,
                            score=game.score,
                            steps=game.steps
                        )
                        
                        # Initialize round count for this game if it doesn't exist
                        current_game = game_count + 1
                        if current_game not in game_round_count:
                            game_round_count[current_game] = 0
                        
                        # Increment the round count for this game
                        game_round_count[current_game] += 1
                        
                        # Log the prompt with sequential round number
                        prompt_filename = f"game{current_game}_round{game_round_count[current_game]}_prompt.txt"
                        save_to_file(prompt, prompts_dir, prompt_filename)
                        
                        # Get next move from LLM, passing the specified model if provided
                        llm_kwargs = {}
                        if args.model:
                            llm_kwargs['model'] = args.model
                        llm_response = llm_client.generate_response(prompt, **llm_kwargs)
                        
                        # Log the response with sequential round number
                        response_filename = f"game{current_game}_round{game_round_count[current_game]}_response.txt"
                        save_to_file(llm_response, responses_dir, response_filename)
                        
                        # Parse and get the first move from the sequence
                        next_move = game.parse_llm_response(llm_response)
                        print(Fore.CYAN + f"🐍 Move: {next_move if next_move else 'None - staying in place'} (Game {current_game}, Round {game_round_count[current_game]})")
                        
                        # We now have a new plan, so don't request another one until we need it
                        need_new_plan = False
                        
                        # Only execute the move if we got a valid direction
                        if next_move:
                            # Execute the move and check if game continues
                            game_active, apple_eaten = game.make_move(next_move)
                        else:
                            # No valid move found, but we still count this as a round
                            print(Fore.YELLOW + "No valid move found in LLM response. Snake stays in place.")
                            # No movement, so the game remains active and no apple is eaten
                            game_active, apple_eaten = True, False
                        
                        # Increment round count (global count across all moves)
                        round_count += 1
                    else:
                        # Get the next move from the existing plan
                        next_move = game.get_next_planned_move()
                        
                        # If we have a move, execute it
                        if next_move:
                            current_game = game_count + 1
                            print(Fore.CYAN + f"🐍 Executing planned move: {next_move} (Game {current_game}, Round {game_round_count[current_game]})")
                            
                            # Execute the move and check if game continues
                            game_active, apple_eaten = game.make_move(next_move)
                            
                            # If we've eaten an apple, request a new plan
                            if apple_eaten:
                                print(Fore.GREEN + "🍎 Apple eaten! Requesting new plan.")
                                need_new_plan = True
                            
                            # We don't increment game_round_count here as we're executing planned moves,
                            # not generating new LLM responses and prompts
                            round_count += 1  # Only increment the global round count
                            
                            # Pause between moves for visualization
                            if not speed_up:
                                time.sleep(args.move_pause)
                        else:
                            # No more planned moves, request a new plan
                            need_new_plan = True
                    
                    # Check if game is over
                    if not game_active:
                        game_count += 1
                        current_game = game_count
                        print(Fore.RED + f"❌ Game over! Score: {game.score}, Steps: {game.steps}")
                        
                        # Save game summary
                        summary = f"""Game {current_game} Summary:
Score: {game.score}
Steps: {game.steps}
Rounds: {game_round_count.get(current_game, 0)}
Last direction: {next_move}
"""
                        save_to_file(summary, log_dir, f"game{current_game}_summary.txt")
                        
                        # The game_round_count dictionary maintains separate round counts per game,
                        # so we don't need to reset it here
                        
                        # Wait a moment before resetting if not the last game
                        if game_count < args.max_games:
                            pygame.time.delay(1000)  # Wait 1 second
                            game.reset()
                            game_active = True
                            need_new_plan = True
                            print(Fore.GREEN + f"🔄 Starting game {game_count + 1}/{args.max_games}")
                    
                    # Update the game and draw
                    game.update()
                    game.draw()
                    
                except Exception as e:
                    print(Fore.RED + f"Error in game loop: {e}")
                    import traceback
                    traceback.print_exc()
                    # Continue to next iteration, don't crash the game
            
            # Control game speed
            pygame.time.delay(time_delay)
            clock.tick(time_tick)
        
        print(Fore.GREEN + f"👋 Game session complete. Played {game_count} games.")
        print(Fore.GREEN + f"💾 Logs saved to {os.path.abspath(log_dir)}")
        
        # Write summary info file
        info_content = f"""Session Summary:
Total games played: {game_count}
Total moves executed: {round_count}
Rounds per game: {', '.join([f'Game {g}: {r}' for g, r in game_round_count.items()])}
"""
        save_to_file(info_content, log_dir, "info.txt")
        
    except Exception as e:
        print(Fore.RED + f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    main()