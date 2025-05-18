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
from config import TIME_DELAY, TIME_TICK
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
                      help='LLM provider to use (hunyuan or ollama)')
    parser.add_argument('--model', type=str, default=None,
                      help='Model name to use when provider is ollama (default: auto-select largest model)')
    parser.add_argument('--max-games', type=int, default=100,
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
        
        # Game variables
        speed_up = False
        time_delay = TIME_DELAY
        time_tick = TIME_TICK
        clock = pygame.time.Clock()
        game_count = 0
        game_active = True
        round_count = 0
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
                    elif event.key == pygame.K_s:
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
                        # Note: The format is now handled directly in get_state_representation()
                        prompt = game_state
                        
                        # Log the prompt
                        prompt_filename = f"game{game_count+1}_round{round_count+1}_prompt.txt"
                        save_to_file(prompt, prompts_dir, prompt_filename)
                        
                        # Get next move from LLM, passing model name if specified and provider is ollama
                        kwargs = {}
                        if args.provider == 'ollama' and args.model:
                            kwargs['model'] = args.model
                            print(Fore.CYAN + f"Using Ollama model: {args.model}")
                        else:
                            print(Fore.CYAN + f"Using default model for provider: {args.provider}")
                            
                        llm_response = llm_client.generate_response(prompt, **kwargs)
                        
                        # Log the response
                        response_filename = f"game{game_count+1}_round{round_count+1}_response.txt"
                        save_to_file(llm_response, responses_dir, response_filename)
                        
                        # Parse and get the first move from the sequence
                        next_move = game.parse_llm_response(llm_response)
                        print(Fore.CYAN + f"🐍 Move: {next_move} (Game {game_count+1}, Round {round_count+1})")
                        
                        # We now have a new plan, so don't request another one until we need it
                        need_new_plan = False
                        
                        # Execute the move and check if game continues
                        game_active, apple_eaten = game.make_move(next_move)
                        
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
                            if not speed_up:
                                time.sleep(args.move_pause)
                        else:
                            # No more planned moves, request a new plan
                            need_new_plan = True
                    
                    # Check if game is over
                    if not game_active:
                        game_count += 1
                        print(Fore.RED + f"❌ Game over! Score: {game.score}, Steps: {game.steps}")
                        
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