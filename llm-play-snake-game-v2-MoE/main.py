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
from llm_parser import LLMOutputParser
from config import TIME_DELAY, TIME_TICK, MOVE_PAUSE
from colorama import Fore, init as init_colorama
from text_utils import (
    save_to_file, 
    save_experiment_info, 
    update_experiment_info, 
    format_raw_llm_response, 
    format_parsed_llm_response,
    generate_game_summary
)

# Initialize colorama for colored terminal output
init_colorama(autoreset=True)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='LLM-guided Snake game')
    parser.add_argument('--provider', type=str, default='hunyuan',
                      help='LLM provider to use (hunyuan, ollama, deepseek, or mistral)')
    parser.add_argument('--model', type=str, default=None,
                      help='Model name to use for first LLM. For Ollama: check first what\'s available on the server. For DeepSeek: "deepseek-chat" or "deepseek-reasoner". For Mistral: "mistral-medium-latest" (default) or "mistral-large-latest"')
    parser.add_argument('--parser-provider', type=str, default=None,
                      help='LLM provider to use for parsing (if not specified, uses the same as --provider)')
    parser.add_argument('--parser-model', type=str, default=None,
                      help='Model name to use for parsing (if not specified, uses the default for the parser provider)')
    parser.add_argument('--max-games', type=int, default=6,
                      help='Maximum number of games to play')
    parser.add_argument('--move-pause', type=float, default=MOVE_PAUSE,
                      help='Pause between sequential moves in seconds (default: 1.0)')
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Validate the command-line arguments to detect duplicate or invalid arguments
    raw_args = ' '.join(sys.argv[1:])
    
    # Check for duplicate --model arguments (which would silently overwrite each other)
    model_count = raw_args.count('--model')
    if model_count > 1:
        raise ValueError(f"Error: '--model' argument appears {model_count} times. Use '--model' for the first LLM and '--parser-model' for the parser LLM.")
    
    # Check for duplicate --provider arguments
    provider_count = raw_args.count('--provider')
    if provider_count > 1:
        raise ValueError(f"Error: '--provider' argument appears {provider_count} times. Use '--provider' for the first LLM and '--parser-provider' for the parser LLM.")
    
    return args

def main():
    """Initialize and run the LLM Snake game."""
    try:
        # Parse command line arguments
        try:
            args = parse_arguments()
        except ValueError as e:
            # Handle command line argument errors cleanly
            print(Fore.RED + f"Command-line error: {e}")
            print(Fore.YELLOW + "For help, use: python main.py --help")
            sys.exit(1)
        
        # Initialize pygame
        pygame.init()
        pygame.font.init()
        
        # Set up the game
        game = SnakeGame()
        
        # Set up the main LLM client for generating moves
        llm_client = LLMClient(provider=args.provider, model=args.model)
        print(Fore.GREEN + f"✅ Using LLM provider for moves: {args.provider}")
        if args.model:
            print(Fore.GREEN + f"✅ Using model for moves: {args.model}")
        
        # Set up the parser LLM client for formatting the response
        parser_provider = args.parser_provider if args.parser_provider else args.provider
        parser_model = args.parser_model
        parser_client = LLMOutputParser(provider=parser_provider, model=parser_model)
        print(Fore.GREEN + f"✅ Using LLM provider for parsing: {parser_provider}")
        if parser_model:
            print(Fore.GREEN + f"✅ Using model for parsing: {parser_model}")
        
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
        parser_usage_count = 0  # Track how many times the parser LLM is used
        previous_parser_usage = 0  # Track previous parser usage
        
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
                        prompt_path = save_to_file(prompt, prompts_dir, prompt_filename)
                        print(Fore.GREEN + f"📝 Prompt saved to {prompt_path}")
                        
                        # Get next move from first LLM
                        kwargs = {}
                        if args.model:
                            kwargs['model'] = args.model
                            print(Fore.CYAN + f"Using {args.provider} model: {args.model}")
                        else:
                            print(Fore.CYAN + f"Using default model for provider: {args.provider}")
                            
                        # Get raw response from first LLM
                        request_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        raw_llm_response = llm_client.generate_response(prompt, **kwargs)
                        response_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        
                        # Format the raw response with timestamp metadata
                        model_name = args.model if args.model else f'Default model for {args.provider}'
                        timestamped_response = format_raw_llm_response(
                            raw_llm_response, 
                            request_time, 
                            response_time, 
                            model_name, 
                            args.provider
                        )
                        
                        # Log the raw response
                        raw_response_filename = f"game{game_count+1}_round{round_count+1}_raw_response.txt"
                        save_to_file(timestamped_response, responses_dir, raw_response_filename)
                        
                        # Use second LLM to parse and format the response
                        print(Fore.CYAN + f"Parsing response with second LLM")
                        parser_request_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        parsed_response, parser_prompt = parser_client.parse_and_format(raw_llm_response)
                        parser_response_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        
                        # Always log the parser prompt and increment usage count
                        parser_usage_count += 1
                        parser_prompt_filename = f"game{game_count+1}_round{round_count+1}_parser_prompt.txt"
                        save_to_file(parser_prompt, prompts_dir, parser_prompt_filename)
                        print(Fore.GREEN + f"📝 Parser prompt saved to {parser_prompt_filename}")
                        
                        # Format the parsed response with timestamp metadata
                        parser_model_name = args.parser_model if args.parser_model else f'Default model for {parser_provider}'
                        timestamped_parsed_response = format_parsed_llm_response(
                            parsed_response, 
                            parser_request_time, 
                            parser_response_time, 
                            parser_model_name, 
                            parser_provider
                        )
                        
                        # Log the parsed response
                        response_filename = f"game{game_count+1}_round{round_count+1}_parsed_response.txt"
                        save_to_file(timestamped_parsed_response, responses_dir, response_filename)
                        
                        # Parse and get the first move from the sequence
                        next_move = game.parse_llm_response(parsed_response)
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
                        
                        # Calculate game-specific statistics
                        game_parser_usage = parser_usage_count if game_count == 1 else parser_usage_count - previous_parser_usage
                        previous_parser_usage = parser_usage_count
                        
                        # Save detailed game summary
                        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        summary = generate_game_summary(
                            game_count,
                            now,
                            game.score,
                            game.steps,
                            next_move,
                            game_parser_usage,
                            len(game.snake_positions),
                            game.last_collision_type,
                            round_count
                        )
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
        update_experiment_info(log_dir, game_count, total_score, total_steps, parser_usage_count)
        
        print(Fore.GREEN + f"👋 Game session complete. Played {game_count} games.")
        print(Fore.GREEN + f"💾 Logs saved to {os.path.abspath(log_dir)}")
        print(Fore.GREEN + f"🔄 Parser LLM was used {parser_usage_count} times")
        
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