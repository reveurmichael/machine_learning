"""
Game manager module for the Snake game.
Handles game session management, initialization, and statistics reporting.
"""

import os
import time
import pygame
import traceback
import json
from datetime import datetime
from colorama import Fore
from core.snake_game import SnakeGame
from gui.game_gui import GameGUI
from llm_client import LLMClient
from llm_parser import LLMOutputParser
from config import TIME_DELAY, TIME_TICK, MOVE_PAUSE
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

from utils.llm_utils import handle_llm_response


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
        self.parser_client = None
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
        self.game = SnakeGame(use_gui=self.use_gui)
        
        # Set up the GUI if needed
        if self.use_gui:
            gui = GameGUI()
            self.game.set_gui(gui)
        
        # Set up the primary LLM client
        self.llm_client = LLMClient(provider=self.args.provider, model=self.args.model)
        print(Fore.GREEN + f"✅ Using primary LLM provider: {self.args.provider}")
        if self.args.model:
            print(Fore.GREEN + f"✅ Using primary LLM model: {self.args.model}")
        
        # Set up the secondary LLM client
        self.parser_provider = self.args.parser_provider if self.args.parser_provider else self.args.provider
        self.parser_model = self.args.parser_model
        
        if self.parser_provider and self.parser_provider.lower() != "none":
            self.parser_client = LLMOutputParser(provider=self.parser_provider, model=self.parser_model)
            print(Fore.GREEN + f"✅ Using parser LLM provider: {self.parser_provider}")
            if self.parser_model:
                print(Fore.GREEN + f"✅ Using parser LLM model: {self.parser_model}")
        else:
            print(Fore.GREEN + f"✅ Not using a parser LLM - primary LLM output will be used directly")
        
        print(Fore.GREEN + f"⏱️ Pause between moves: {self.args.move_pause} seconds")
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
        
        # Calculate and record response time duration
        primary_response_duration = (response_timestamp - request_timestamp).total_seconds()
        self.game.add_response_time(primary_response_duration)
        
        # Format the raw response with timestamp metadata
        model_name = self.args.model if self.args.model else f'Default model for {self.args.provider}'
        timestamped_response = format_raw_llm_response(
            raw_llm_response, 
            request_time, 
            response_time, 
            model_name, 
            self.args.provider,
            parser_model=self.args.parser_model,
            parser_provider=self.args.parser_provider,
            response_duration=primary_response_duration
        )
        
        # Log the raw response from primary LLM
        raw_response_filename = f"game{self.game_count+1}_round{self.round_count+1}_raw_response.txt"
        save_to_file(timestamped_response, self.responses_dir, raw_response_filename)
        
        # Check if we should use the parser LLM
        if self.parser_provider and self.parser_provider.lower() != "none":
            # Get the current head and apple positions for the parser
            head_x, head_y = self.game.head_position
            head_pos = f"({head_x}, {head_y})"
            apple_x, apple_y = self.game.apple_position
            apple_pos = f"({apple_x}, {apple_y})"
            
            # Get body cells for the parser using the helper method
            body_cells_str = self.game.format_body_cells_str(self.game.snake_positions)
            
            # Use secondary LLM to parse and format the response with timing
            print(Fore.CYAN + f"Using secondary LLM to parse primary LLM's response")
            parser_request_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            parser_request_timestamp = datetime.now()
            parsed_response, parser_prompt = self.parser_client.parse_and_format(
                raw_llm_response, 
                head_pos=head_pos, 
                apple_pos=apple_pos,
                body_cells=body_cells_str
            )
            parser_response_timestamp = datetime.now()
            parser_response_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Calculate and record secondary LLM response time
            secondary_response_duration = (parser_response_timestamp - parser_request_timestamp).total_seconds()
            self.game.add_secondary_response_time(secondary_response_duration)
            
            # Log the parser prompt and increment usage count
            self.parser_usage_count += 1
            parser_prompt_filename = f"game{self.game_count+1}_round{self.round_count+1}_parser_prompt.txt"
            save_to_file(parser_prompt, self.prompts_dir, parser_prompt_filename)
            print(Fore.GREEN + f"📝 Parser prompt saved to {parser_prompt_filename}")
            
            # Format the parsed response with timestamp metadata
            parser_model_name = self.args.parser_model if self.args.parser_model else f'Default model for {self.parser_provider}'
            timestamped_parsed_response = format_parsed_llm_response(
                parsed_response, 
                parser_request_time, 
                parser_response_time, 
                parser_model_name, 
                self.parser_provider,
                response_duration=secondary_response_duration
            )
            
            # Log the parsed response from secondary LLM
            response_filename = f"game{self.game_count+1}_round{self.round_count+1}_parsed_response.txt"
            save_to_file(timestamped_parsed_response, self.responses_dir, response_filename)
            
            # Parse and get the first move from the sequence
            next_move = self.game.parse_llm_response(parsed_response)
            
            # Handle the response and update counters
            self.error_steps, self.empty_steps, self.consecutive_empty_steps, game_active = handle_llm_response(
                parsed_response, next_move, self.error_steps, self.empty_steps, 
                self.consecutive_empty_steps, self.args.max_empty_moves
            )
            
            # Update collision type if needed
            if not game_active:
                self.game.last_collision_type = 'empty_moves'
        else:
            # Use the primary LLM response directly
            print(Fore.CYAN + f"Using primary LLM response directly (no parser)")
            next_move = self.game.parse_llm_response(raw_llm_response)
            # No parser usage in this case
            
            # Handle the response and update counters
            self.error_steps, self.empty_steps, self.consecutive_empty_steps, game_active = handle_llm_response(
                raw_llm_response, next_move, self.error_steps, self.empty_steps, 
                self.consecutive_empty_steps, self.args.max_empty_moves
            )
            
            # Update collision type if needed
            if not game_active:
                self.game.last_collision_type = 'empty_moves'
        
        print(Fore.CYAN + f"🐍 Move: {next_move if next_move else 'None - staying in place'} (Game {self.game_count+1}, Round {self.round_count+1})")
        
        # Record the move if valid
        if next_move:
            self.current_game_moves.append(next_move)
        
        return next_move, game_active
    
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
        # Use the utility function from utils
        self.game_count, self.total_score, self.total_steps, self.game_scores, self.round_count, self.previous_parser_usage = process_game_over(
            self.game, self.game_count, self.total_score, self.total_steps, 
            self.game_scores, self.round_count, self.parser_usage_count, 
            self.previous_parser_usage, self.log_dir, self.args, self.current_game_moves
        )
        
        # Reset for next game
        self.current_game_moves = []  # Reset moves for next game
        
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
        # Use the utility function from utils
        report_final_statistics(
            self.log_dir, self.game_count, self.total_score, self.total_steps,
            self.parser_usage_count, self.game_scores, self.empty_steps, 
            self.error_steps, self.args.max_empty_moves
        )
    
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
                                time.sleep(self.args.move_pause)
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