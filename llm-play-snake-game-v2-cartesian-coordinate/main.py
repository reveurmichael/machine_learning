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
from config import TIME_DELAY, TIME_TICK, MOVE_PAUSE
from colorama import Fore, init as init_colorama

# Initialize colorama for colored terminal output
init_colorama(autoreset=True)

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

def save_experiment_info(args, directory):
    """Save experiment information to a file.
    
    Args:
        args: Command line arguments
        directory: Directory to save to
        
    Returns:
        Path to the saved file
    """
    # Create content with experiment information
    content = f"""Experiment Information
====================
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

LLM Provider: {args.provider}
Model: {args.model if args.model else 'Default model for provider'}
Max Games: {args.max_games}
Move Pause: {args.move_pause} seconds

Other Information:
- Time Delay: {TIME_DELAY}
- Time Tick: {TIME_TICK}
"""
    
    # Save to file
    return save_to_file(content, directory, "info.txt")

def update_experiment_info(directory, game_count, total_score, total_steps):
    """Update the experiment information file with game statistics.
    
    Args:
        directory: Directory containing the info.txt file
        game_count: Total number of games played
        total_score: Total score across all games
        total_steps: Total steps taken across all games
    """
    file_path = os.path.join(directory, "info.txt")
    
    # Read existing content
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Add statistics section
    stats = f"""

Game Statistics
==============
Total Games Played: {game_count}
Total Score: {total_score}
Total Steps: {total_steps}
Average Score per Game: {total_score/game_count:.2f}
Average Steps per Game: {total_steps/game_count:.2f}
"""
    
    # Append statistics to content
    content += stats
    
    # Write updated content back to file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='LLM-guided Snake game')
    parser.add_argument('--provider', type=str, default='hunyuan',
                      help='LLM provider to use (hunyuan, ollama, deepseek, or mistral)')
    parser.add_argument('--model', type=str, default=None,
                      help='Model name to use. For Ollama, check first what\'s available on the server. For DeepSeek: "deepseek-chat" or "deepseek-reasoner". For Mistral: "mistral-medium-latest" (default) or "mistral-large-latest"')
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
        print(Fore.GREEN + f"⏱️ Pause between moves: {args.move_pause} seconds")
        
        # Set up logging directories
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = f"game_logs_{timestamp}"
        prompts_dir = os.path.join(log_dir, "prompts")
        responses_dir = os.path.join(log_dir, "responses")
        
        # Save experiment information
        model_info_path = save_experiment_info(args, log_dir)
        print(Fore.GREEN + f"📝 Experiment information saved to {model_info_path}")
        
        # Game variables
        time_delay = TIME_DELAY
        time_tick = TIME_TICK
        clock = pygame.time.Clock()
        game_count = 0
        game_active = True
        round_count = 0
        need_new_plan = True
        total_score = 0
        total_steps = 0
        
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
                        # Note: The format is now handled directly in get_state_representation()
                        prompt = game_state
                        
                        # Log the prompt
                        prompt_filename = f"game{game_count+1}_round{round_count+1}_prompt.txt"
                        save_to_file(prompt, prompts_dir, prompt_filename)
                        
                        # Get next move from LLM, passing model name if specified and provider is ollama
                        kwargs = {}
                        if args.model:
                            kwargs['model'] = args.model
                            print(Fore.CYAN + f"Using {args.provider} model: {args.model}")
                        else:
                            print(Fore.CYAN + f"Using default model for provider: {args.provider}")
                            
                        llm_response = llm_client.generate_response(prompt, **kwargs)
                        
                        # Log the response
                        response_filename = f"game{game_count+1}_round{round_count+1}_response.txt"
                        save_to_file(llm_response, responses_dir, response_filename)
                        
                        # Parse and get the first move from the sequence
                        next_move = game.parse_llm_response(llm_response)
                        print(Fore.CYAN + f"🐍 Move: {next_move if next_move else 'None - staying in place'} (Game {game_count+1}, Round {round_count+1})")
                        
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
                        
                        # Increment round count
                        round_count += 1
                    else:
                        # Get the next move from the existing plan
                        next_move = game.get_next_planned_move()
                        
                        # If we have a move, execute it
                        if next_move:
                            print(Fore.CYAN + f"🐍 Executing planned move: {next_move} (Game {game_count+1}, Round {round_count+1})")
                            
                            # Execute the move and check if game continues
                            game_active, apple_eaten = game.make_move(next_move)
                            
                            # If we've eaten an apple, request a new plan
                            if apple_eaten:
                                print(Fore.GREEN + f"🍎 Apple eaten! Requesting new plan.")
                                need_new_plan = True
                            
                            # Increment round count
                            round_count += 1
                            
                            # Pause between moves for visualization
                            time.sleep(args.move_pause)
                        else:
                            # No more planned moves, request a new plan
                            need_new_plan = True
                    
                    # Check if game is over
                    if not game_active:
                        game_count += 1
                        print(Fore.RED + f"❌ Game over! Score: {game.score}, Steps: {game.steps}")
                        
                        # Update totals
                        total_score += game.score
                        total_steps += game.steps
                        
                        # Save game summary
                        summary = f"""Game {game_count} Summary:
Score: {game.score}
Steps: {game.steps}
Last direction: {next_move}
"""
                        save_to_file(summary, log_dir, f"game{game_count}_summary.txt")
                        
                        # Reset round count for next game
                        round_count = 0
                        
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
        
        # Update experiment info with final statistics
        update_experiment_info(log_dir, game_count, total_score, total_steps)
        
        print(Fore.GREEN + f"👋 Game session complete. Played {game_count} games.")
        print(Fore.GREEN + f"💾 Logs saved to {os.path.abspath(log_dir)}")
        
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
