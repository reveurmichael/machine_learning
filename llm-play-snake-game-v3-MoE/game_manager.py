"""
Game manager module for the Snake game.
Handles game session management, initialization, and statistics reporting.
"""

import os
import time
import pygame
import traceback
import json
import sys
from datetime import datetime
from colorama import Fore
from core.game_logic import GameLogic
from gui.game_gui import GameGUI
from llm_client import LLMClient
from config import TIME_DELAY, TIME_TICK, PAUSE_BETWEEN_MOVES_SECONDS
from utils import (
    # Log utilities
    save_to_file,
    format_raw_llm_response,
    format_parsed_llm_response,
    generate_game_summary_json,
    
    # JSON utilities
    get_json_error_stats,
    reset_json_error_stats,
    save_experiment_info_json,
    update_experiment_info_json,
    
    # Game management utilities
    check_max_steps,
    process_game_over,
    handle_error,
    report_final_statistics
)

from utils.llm_utils import handle_llm_response, check_llm_health, parse_and_format


class GameManager:
    """Manages the overall game session, including multiple games and statistics."""
    
    def __init__(self, args):
        """Initialize the game manager.
        
        Args:
            args: Command line arguments
        """
        self.args = args
        
        # Game counters and statistics
        self.game_count = 0
        self.round_count = 0
        self.total_score = 0
        self.total_steps = 0
        self.empty_steps = 0
        self.error_steps = 0
        self.consecutive_empty_steps = 0
        self.game_scores = []
        self.parser_usage_count = 0
        self.previous_parser_usage = 0
        
        # Game state
        self.game = None
        self.game_active = True
        self.need_new_plan = True
        self.running = True
        
        # Track moves for this game
        self.current_game_moves = []
        
        # Pygame and timing
        self.clock = pygame.time.Clock()
        self.time_delay = TIME_DELAY
        self.time_tick = TIME_TICK
        
        # LLM clients
        self.llm_client = None
        self.parser_provider = None
        self.parser_model = None
        
        # Logging directories
        self.log_dir = None
        self.prompts_dir = None
        self.responses_dir = None
        
        # GUI settings
        self.use_gui = not args.no_gui
    
    def initialize(self):
        """Initialize the game, LLM clients, and logging directories."""
        # Reset JSON error statistics
        reset_json_error_stats()
        
        # Initialize primary LLM client for health check
        self.llm_client = LLMClient(provider=self.args.provider, model=self.args.model)
        print(Fore.GREEN + f"✅ Using primary LLM provider: {self.args.provider}")
        if self.args.model:
            print(Fore.GREEN + f"✅ Using primary LLM model: {self.args.model}")
            
        # Configure secondary LLM (parser) if specified
        if self.args.parser_provider and self.args.parser_provider.lower() != "none":
            print(Fore.GREEN + f"✅ Using parser LLM provider: {self.args.parser_provider}")
            parser_model = self.args.parser_model
            print(Fore.GREEN + f"✅ Using parser LLM model: {parser_model}")
            
            # Set up the secondary LLM in the client
            self.llm_client.set_secondary_llm(self.args.parser_provider, parser_model)
            
            # Perform health check for parser LLM
            parser_healthy, _ = check_llm_health(
                LLMClient(provider=self.args.parser_provider, model=parser_model)
            )
            if not parser_healthy:
                print(Fore.RED + f"❌ Parser LLM health check failed. Continuing without parser.")
                self.args.parser_provider = "none"
                self.args.parser_model = None
        else:
            print(Fore.YELLOW + "⚠️ No parser LLM specified. Using primary LLM output directly.")
            self.args.parser_provider = "none"
            self.args.parser_model = None
        
        # Handle sleep before launching if specified
        if self.args.sleep_before_launching > 0:
            minutes = self.args.sleep_before_launching
            print(Fore.YELLOW + f"💤 Sleeping for {minutes} minute{'s' if minutes > 1 else ''} before launching...")
            time.sleep(minutes * 60)
            print(Fore.GREEN + "⏰ Waking up and starting the program...")
        
        # Initialize pygame if using GUI
        if self.use_gui:
            pygame.init()
            pygame.font.init()
        
        # Set up the game
        self.game = GameLogic(use_gui=self.use_gui)
        
        # Set up the GUI if needed
        if self.use_gui:
            gui = GameGUI()
            self.game.set_gui(gui)
        
        print(Fore.GREEN + f"⏱️ Pause between moves: {PAUSE_BETWEEN_MOVES_SECONDS} seconds")
        print(Fore.GREEN + f"⏱️ Maximum steps per game: {self.args.max_steps}")
        
        # Set up logging directories
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        primary_model = self.args.model if self.args.model else f'default_{self.args.provider}'
        primary_model = primary_model.replace(':', '-')  # Replace colon with hyphen
        self.log_dir = f"{primary_model}_{timestamp}"
        self.prompts_dir = os.path.join(self.log_dir, "prompts")
        self.responses_dir = os.path.join(self.log_dir, "responses")
        
        # Save experiment information
        model_info_path = save_experiment_info_json(self.args, self.log_dir)
        print(Fore.GREEN + f"📝 Experiment information saved to {model_info_path}")
    
    def process_events(self):
        """Process pygame events."""
        if not self.use_gui:
            return
            
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_r:
                    # Reset game
                    self.game.reset()
                    self.game_active = True
                    self.need_new_plan = True
                    self.consecutive_empty_steps = 0  # Reset on game reset
                    self.current_game_moves = []  # Reset moves for new game
                    print(Fore.GREEN + "🔄 Game reset")
    
    def get_llm_response(self):
        """Get a response from the LLM based on the current game state.
        
        Returns:
            Tuple of (next_move, game_active)
        """
        # Get game state
        game_state = self.game.get_state_representation()
        
        # Format prompt for LLM
        prompt = game_state
        
        # Log the prompt
        prompt_filename = f"game{self.game_count+1}_round{self.round_count+1}_prompt.txt"
        prompt_path = save_to_file(prompt, self.prompts_dir, prompt_filename)
        print(Fore.GREEN + f"📝 Prompt saved to {prompt_path}")
        
        # Get next move from first LLM
        kwargs = {}
        if self.args.model:
            kwargs['model'] = self.args.model
            print(Fore.CYAN + f"Using {self.args.provider} model: {self.args.model}")
        else:
            print(Fore.CYAN + f"Using default model for provider: {self.args.provider}")
            
        # Get raw response from first LLM with timing
        request_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        request_timestamp = datetime.now()
        raw_llm_response = self.llm_client.generate_response(prompt, **kwargs)
        response_timestamp = datetime.now()
        response_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Record primary response time
        primary_response_time = (response_timestamp - request_timestamp).total_seconds()
        self.game.game_state.record_primary_response_time(primary_response_time)
        
        # Record primary token statistics if available
        if hasattr(self.llm_client, 'last_token_count') and self.llm_client.last_token_count:
            token_count = self.llm_client.last_token_count
            if 'prompt_tokens' in token_count and 'completion_tokens' in token_count:
                self.game.game_state.record_primary_token_stats(
                    token_count['prompt_tokens'],
                    token_count['completion_tokens']
                )
        
        # Extract necessary information for parser
        head_pos, apple_pos, body_cells = self._extract_state_for_parser()
        
        # Process the response
        if self.parser_provider and self.parser_provider.lower() != "none":
            print(Fore.CYAN + f"Using {self.parser_provider} LLM to parse response")
            parser_client = LLMClient(provider=self.parser_provider, model=self.parser_model)
            
            # Format the response using the secondary LLM
            self.parser_usage_count += 1
            parser_request_timestamp = datetime.now()
            formatted_response, parser_prompt = parse_and_format(
                parser_client, 
                raw_llm_response,
                head_pos=head_pos,
                apple_pos=apple_pos,
                body_cells=body_cells
            )
            parser_response_timestamp = datetime.now()
            
            # Record secondary response time
            secondary_response_time = (parser_response_timestamp - parser_request_timestamp).total_seconds()
            self.game.game_state.record_secondary_response_time(secondary_response_time)
            
            # Record secondary token statistics if available
            if hasattr(parser_client, 'last_token_count') and parser_client.last_token_count:
                token_count = parser_client.last_token_count
                if 'prompt_tokens' in token_count and 'completion_tokens' in token_count:
                    self.game.game_state.record_secondary_token_stats(
                        token_count['prompt_tokens'],
                        token_count['completion_tokens']
                    )
            
            # Log the parser prompt
            parser_prompt_filename = f"game{self.game_count+1}_round{self.round_count+1}_parser_prompt.txt"
            parser_prompt_path = save_to_file(parser_prompt, self.prompts_dir, parser_prompt_filename)
            
            # Calculate timing information
            parsing_time = (datetime.now() - response_timestamp).total_seconds()
            
            # Log the response and timing information
            response_filename = f"game{self.game_count+1}_round{self.round_count+1}_response.txt"
            response_path = save_to_file(
                format_parsed_llm_response(
                    raw_llm_response, 
                    formatted_response, 
                    request_time, 
                    response_time,
                    parsing_time
                ),
                self.responses_dir, 
                response_filename
            )
            print(Fore.GREEN + f"📝 Response saved to {response_path}")
            
            # Parse the response to get the next move
            from utils.json_utils import extract_valid_json
            json_data = extract_valid_json(formatted_response)
            if not json_data or "moves" not in json_data:
                next_move = None
            else:
                moves = json_data["moves"]
                next_move = moves[0] if moves and len(moves) > 0 else None
                
                # Save remaining moves for future use
                if moves and len(moves) > 1:
                    self.game.planned_moves = moves[1:]
                else:
                    self.game.planned_moves = []
        else:
            # Use the raw response directly
            formatted_response = raw_llm_response
            
            # Log the response
            response_filename = f"game{self.game_count+1}_round{self.round_count+1}_response.txt"
            response_path = save_to_file(
                format_raw_llm_response(
                    raw_llm_response, 
                    request_time, 
                    response_time,
                    model_name=self.args.model,
                    provider=self.args.provider
                ),
                self.responses_dir, 
                response_filename
            )
            print(Fore.GREEN + f"📝 Response saved to {response_path}")
            
            # Parse the response to get the next move
            from utils.json_utils import extract_valid_json
            json_data = extract_valid_json(formatted_response)
            if not json_data or "moves" not in json_data:
                next_move = None
            else:
                moves = json_data["moves"]
                next_move = moves[0] if moves and len(moves) > 0 else None
                
                # Save remaining moves for future use
                if moves and len(moves) > 1:
                    self.game.planned_moves = moves[1:]
                else:
                    self.game.planned_moves = []
        
        # Process the next move
        self.error_steps, self.empty_steps, self.consecutive_empty_steps, game_active = handle_llm_response(
            formatted_response,
            next_move,
            self.error_steps,
            self.empty_steps,
            self.consecutive_empty_steps,
            self.args.max_empty_moves
        )
        
        # Update the current game moves
        if next_move:
            self.current_game_moves.append(next_move)
            
        return next_move, game_active
    
    def _extract_state_for_parser(self):
        """Extract state information for the parser.
        
        Returns:
            Tuple of (head_pos, apple_pos, body_cells) as strings
        """
        # Get the game state
        head_x, head_y = self.game.head
        apple_x, apple_y = self.game.apple
        body_cells = self.game.body
        
        # Format for parser
        head_pos = f"({head_x}, {head_y})"
        apple_pos = f"({apple_x}, {apple_y})"
        body_cells_str = str(body_cells) if body_cells else "[]"
        
        return head_pos, apple_pos, body_cells_str
    
    def check_max_steps(self):
        """Check if the game has reached the maximum number of steps.
        
        Returns:
            Boolean indicating if max steps has been reached
        """
        # Use the utility function from utils
        return check_max_steps(self.game, self.args.max_steps)
    
    def process_game_over(self, next_move=None):
        """Process game over state and prepare for the next game.
        
        Args:
            next_move: The last move made (or None)
        """
        # Update game count and statistics
        self.game_count += 1
        self.total_score += self.game.score
        self.total_steps += self.game.steps
        self.game_scores.append(self.game.score)
        
        # Set a reason if not already set by the game engine
        if not self.game.game_state.game_end_reason:
            if self.game.last_collision_type == 'empty_moves':
                self.game.game_state.record_game_end("EMPTY_MOVES")
            elif self.game.last_collision_type == 'max_steps':
                self.game.game_state.record_game_end("MAX_STEPS")
            else:
                self.game.game_state.record_game_end("UNKNOWN")
        
        # Save game summary
        game_summary_file = os.path.join(self.log_dir, f"game{self.game_count}.json")
        self.game.game_state.save_game_summary(
            game_summary_file,
            self.args.provider, 
            self.args.model or f"default_{self.args.provider}",
            self.args.parser_provider or self.args.provider,
            self.args.parser_model
        )
        print(Fore.GREEN + f"📊 Game {self.game_count} summary saved to {game_summary_file}")
        
        # Print game over message with statistics
        print(Fore.RED + f"❌ Game over! Score: {self.game.score}, Steps: {self.game.steps}")
        print(Fore.YELLOW + f"Average Score: {self.total_score / self.game_count:.2f} after {self.game_count} games")
        
        # Prepare parser usage tracking for next game
        self.previous_parser_usage = self.parser_usage_count
        
        # Reset round count for next game
        self.round_count = 0
        
        # Reset the current game moves
        self.current_game_moves = []
        
        # Wait a moment before resetting if not the last game
        if self.game_count < self.args.max_games:
            pygame.time.delay(1000)  # Wait 1 second
            self.game.reset()
            self.game_active = True
            self.need_new_plan = True
            print(Fore.GREEN + f"🔄 Starting game {self.game_count + 1}/{self.args.max_games}")
    
    def handle_error(self, error):
        """Handle errors that occur during the game loop.
        
        Args:
            error: The exception that occurred
        """
        # Use the utility function from utils
        from utils.game_manager_utils import handle_error
        self.game_active, self.game_count, self.total_score, self.total_steps, self.game_scores, self.round_count, self.previous_parser_usage = handle_error(
            self.game, self.game_active, self.game_count, self.total_score, self.total_steps, 
            self.game_scores, self.round_count, self.parser_usage_count, self.previous_parser_usage, 
            self.log_dir, self.args, self.current_game_moves, error
        )
        
        # Prepare for next game if we haven't reached the limit
        if self.game_count < self.args.max_games and not self.game_active:
            pygame.time.delay(1000)  # Wait 1 second
            self.game.reset()
            self.game_active = True
            self.need_new_plan = True
            self.current_game_moves = []  # Reset moves for next game
            print(Fore.GREEN + f"🔄 Starting game {self.game_count + 1}/{self.args.max_games}")
    
    def report_final_statistics(self):
        """Report final statistics at the end of the game session."""
        # Only report if games were played
        if self.game_count == 0:
            return
        
        # Get aggregated stats from GameState
        aggregated_stats = self.game.game_state.get_aggregated_stats_for_summary_json(
            self.game_count, 
            self.game_scores
        )
        
        # Add experiment configuration
        aggregated_stats["game_configuration"] = {
            "max_steps_per_game": self.args.max_steps,
            "max_consecutive_empty_moves": self.args.max_empty_moves,
            "max_games": self.args.max_games
        }
        
        # Add LLM usage stats
        aggregated_stats["llm_usage_stats"] = {
            "parser_usage_count": self.parser_usage_count,
            "parser_usage_per_game": self.parser_usage_count / max(1, self.game_count)
        }
        
        # Add efficiency metrics
        aggregated_stats["efficiency_metrics"] = {
            "apples_per_step": self.total_score / max(1, self.total_steps),
            "steps_per_game": self.total_steps / max(1, self.game_count),
            "valid_move_ratio": self.game.game_state.valid_steps / max(1, self.total_steps) * 100
        }
        
        # Update experiment info JSON
        update_experiment_info_json(
            self.log_dir,
            **aggregated_stats,
            json_error_stats=get_json_error_stats()
        )
        
        # Print summary to console
        print(Fore.YELLOW + "\n📊 Final Statistics:")
        print(f"Games Played: {self.game_count}")
        print(f"Total Score: {self.total_score}")
        print(f"Average Score: {aggregated_stats['game_statistics']['mean_score']:.2f}")
        print(f"Best Score: {aggregated_stats['game_statistics']['max_score']}")
        print(f"Steps per Apple: {aggregated_stats['game_statistics']['steps_per_apple']:.2f}")
        print(f"Total Steps: {self.total_steps}")
        
        # Avoid division by zero
        if self.total_steps > 0:
            empty_steps_percent = (self.empty_steps / self.total_steps * 100)
            error_steps_percent = (self.error_steps / self.total_steps * 100)
        else:
            empty_steps_percent = 0
            error_steps_percent = 0
        
        print(f"Empty Steps: {self.empty_steps} ({empty_steps_percent:.2f}%)")
        print(f"Error Steps: {self.error_steps} ({error_steps_percent:.2f}%)")
        
        print(Fore.GREEN + "\n✅ Experiment completed successfully.")
    
    def run_game_loop(self):
        """Run the main game loop."""
        try:
            while self.running and self.game_count < self.args.max_games:
                # Handle events
                self.process_events()
                
                if self.game_active:
                    try:
                        # Check if we need to request a new plan from the LLM
                        if self.need_new_plan:
                            next_move, self.game_active = self.get_llm_response()
                            
                            # We now have a new plan, so don't request another one until we need it
                            self.need_new_plan = False
                            
                            # Check if we've reached max steps
                            if self.check_max_steps():
                                self.game_active = False
                                self.game.game_state.record_game_end("MAX_STEPS")
                            # Only execute the move if we got a valid direction and game is still active
                            elif next_move and self.game_active:
                                # Execute the move and check if game continues
                                self.game_active, apple_eaten = self.game.make_move(next_move)
                            else:
                                # No valid move found, but we still count this as a round
                                print(Fore.YELLOW + "No valid move found in LLM response. Snake stays in place.")
                                # No movement, so the game remains active and no apple is eaten
                                self.game.steps += 1
                                self.total_steps += 1
                                self.game.game_state.record_empty_move()
                            
                            # Increment round count
                            self.round_count += 1
                        else:
                            # Get the next move from the existing plan
                            next_move = self.game.get_next_planned_move()
                            
                            # If we have a move, execute it
                            if next_move:
                                print(Fore.CYAN + f"🐍 Executing planned move: {next_move} (Game {self.game_count+1}, Round {self.round_count+1})")
                                
                                # Record the move
                                self.current_game_moves.append(next_move)
                                
                                # Check if we've reached max steps
                                if self.check_max_steps():
                                    self.game_active = False
                                    self.game.game_state.record_game_end("MAX_STEPS")
                                else:
                                    # Execute the move and check if game continues
                                    self.game_active, apple_eaten = self.game.make_move(next_move)
                                
                                # If we've eaten an apple, request a new plan
                                if apple_eaten:
                                    print(Fore.GREEN + f"🍎 Apple eaten! Requesting new plan.")
                                    self.need_new_plan = True
                                
                                # Increment round count
                                self.round_count += 1
                                
                                # Pause between moves for visualization
                                time.sleep(PAUSE_BETWEEN_MOVES_SECONDS)
                            else:
                                # No more planned moves, request a new plan
                                self.need_new_plan = True
                        
                        # Check if game is over
                        if not self.game_active:
                            self.process_game_over(next_move)
                        
                        # Draw the current state
                        self.game.draw()
                        
                    except Exception as e:
                        self.handle_error(e)
                
                # Control game speed
                pygame.time.delay(self.time_delay)
                self.clock.tick(self.time_tick)
            
            # Report final statistics
            self.report_final_statistics()
            
        except Exception as e:
            print(Fore.RED + f"Fatal error: {e}")
            traceback.print_exc()
        finally:
            # Clean up
            pygame.quit()
    
    def run(self):
        """Initialize and run the game session."""
        try:
            # Initialize the game and LLM clients
            self.initialize()
            
            # Run the game loop
            self.run_game_loop()
            
        except Exception as e:
            # Handle any unexpected errors
            self.handle_error(e)
            
        finally:
            # Final cleanup
            if self.use_gui and pygame.get_init():
                pygame.quit()
            
            # Report final statistics
            self.report_final_statistics()
            
            # Update experiment info with final statistics
            update_experiment_info_json(
                self.log_dir,
                game_count=self.game_count,
                total_score=self.total_score,
                avg_score=self.total_score / max(1, self.game_count),
                total_steps=self.total_steps,
                avg_steps=self.total_steps / max(1, self.game_count),
                json_error_stats=get_json_error_stats()
            ) 