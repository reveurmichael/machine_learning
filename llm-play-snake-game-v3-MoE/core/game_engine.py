"""
Game engine module for the Snake game.
Handles the game loop and move execution.
"""

import os
import json
import time
import pygame
import datetime
import traceback
from colorama import Fore
from llm_client import LLMClient, LLMOutputParser
from utils.log_utils import save_to_file, generate_game_summary_json
from config import PROMPT_TEMPLATE_TEXT


class GameEngine:
    """Handles the game loop and move execution."""

    def __init__(self, game, llm_client, parser_client, move_delay=0.5, max_steps=100, max_empty_moves=3):
        """Initialize the game engine.
        
        Args:
            game: SnakeGame instance
            llm_client: Primary LLM client
            parser_client: Parser LLM client
            move_delay: Delay between moves in seconds
            max_steps: Maximum number of steps per game
            max_empty_moves: Maximum number of empty moves before game over
        """
        self.game = game
        self.llm_client = llm_client
        self.parser_client = parser_client
        self.move_delay = move_delay
        self.max_steps = max_steps
        self.max_empty_moves = max_empty_moves
        
        # Game state
        self.need_new_plan = True
        self.current_plan = []
        self.empty_moves = 0
        self.parser_usage_count = 0
        self.empty_steps = 0
        self.error_steps = 0
        
        # Logging
        self.log_dir = None
        self.prompts_dir = None
        self.responses_dir = None
        
        # Statistics tracking
        self.primary_response_times = []
        self.secondary_response_times = []
        self.apple_positions = []
        self.move_history = []
        self.current_round_moves = []
        self.rounds_data = {}
        self.round_count = 0
        
        # Add the first apple position
        if game and game.apple:
            self.apple_positions.append(game.apple)
            self.round_count = 1

    def set_game(self, game):
        """Set the game instance.
        
        Args:
            game: SnakeGame instance
        """
        self.game = game
        self.need_new_plan = True
        self.current_plan = []
        self.empty_moves = 0
        
        # Reset statistics for the new game
        self.primary_response_times = []
        self.secondary_response_times = []
        self.apple_positions = []
        self.move_history = []
        self.current_round_moves = []
        self.rounds_data = {}
        self.round_count = 0
        
        # Add the first apple position
        if game and game.apple:
            self.apple_positions.append(game.apple)
            self.round_count = 1

    def get_llm_response(self):
        """Get a response from the LLM.
        
        Returns:
            Tuple of (parsed_response, success)
        """
        # Create the prompt
        prompt = self._create_prompt()
        
        # Save the prompt
        if self.prompts_dir:
            prompt_file = os.path.join(
                self.prompts_dir,
                f"prompt_{self.game.steps}.txt"
            )
            save_to_file(prompt, prompt_file)
        
        # Get response from primary LLM with timing
        primary_start_time = time.time()
        response = self.llm_client.generate_response(prompt)
        primary_response_time = time.time() - primary_start_time
        self.primary_response_times.append(primary_response_time)
        
        # Get primary token statistics
        primary_token_stats = self.llm_client.get_token_stats()
        
        # Save the response
        if self.responses_dir:
            response_file = os.path.join(
                self.responses_dir,
                f"response_{self.game.steps}.txt"
            )
            save_to_file(response, response_file)
        
        # Parse the response with timing
        secondary_start_time = time.time()
        parsed_response, success = self.parser_client.parse_and_format(
            response,
            self.game.head,
            self.game.apple,
            self.game.body
        )
        secondary_response_time = time.time() - secondary_start_time
        self.secondary_response_times.append(secondary_response_time)
        
        # Get secondary token statistics
        secondary_token_stats = self.parser_client.get_token_stats()
        
        # Save the response time and token data in the current round
        if self.round_count > 0:
            round_key = f"round_{self.round_count}"
            if round_key not in self.rounds_data:
                self.rounds_data[round_key] = {
                    "apple_position": self.game.apple,
                    "moves": [],
                    "primary_response_times": [],
                    "secondary_response_times": [],
                    "primary_token_stats": [],
                    "secondary_token_stats": []
                }
            
            self.rounds_data[round_key]["primary_response_times"].append(primary_response_time)
            self.rounds_data[round_key]["secondary_response_times"].append(secondary_response_time)
            self.rounds_data[round_key]["primary_token_stats"].append({
                "prompt_tokens": self.llm_client.last_prompt_tokens,
                "completion_tokens": self.llm_client.last_completion_tokens,
                "total_tokens": self.llm_client.last_total_tokens
            })
            self.rounds_data[round_key]["secondary_token_stats"].append({
                "prompt_tokens": self.parser_client.last_prompt_tokens,
                "completion_tokens": self.parser_client.last_completion_tokens,
                "total_tokens": self.parser_client.last_total_tokens
            })
        
        # Save the parser prompt and response
        if self.prompts_dir:
            parser_prompt_file = os.path.join(
                self.prompts_dir,
                f"parser_prompt_{self.game.steps}.txt"
            )
            save_to_file(self.parser_client._create_parser_prompt(
                response, self.game.head, self.game.apple, self.game.body), 
                parser_prompt_file
            )
            
        if self.responses_dir:
            parsed_response_file = os.path.join(
                self.responses_dir,
                f"parsed_response_{self.game.steps}.txt"
            )
            save_to_file(parsed_response, parsed_response_file)
        
        return parsed_response, success

    def _create_prompt(self):
        """Create a prompt for the LLM.
        
        Returns:
            The prompt string
        """
        # Use the game's state representation method
        return self.game.get_state_representation()

    def run_game(self):
        """Run the game loop."""
        # Set up logging directories if they don't exist
        if self.log_dir is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_dir = os.path.join("logs", f"game_{timestamp}")
            
        self.prompts_dir = os.path.join(self.log_dir, "prompts")
        self.responses_dir = os.path.join(self.log_dir, "responses")
        os.makedirs(self.prompts_dir, exist_ok=True)
        os.makedirs(self.responses_dir, exist_ok=True)
        
        # Main game loop
        while not self.game.game_over and self.game.steps < self.max_steps:
            # Process events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return
            
            # Get new plan if needed
            if self.need_new_plan:
                try:
                    # Get response from LLM
                    parsed_response, success = self.get_llm_response()
                    self.parser_usage_count += 1
                    
                    # Log the parser prompt and response
                    print(f"Step {self.game.steps}: Got LLM response, attempting to parse...")
                    
                    # Parse the moves
                    try:
                        # Attempt to parse JSON
                        parsed_json = json.loads(parsed_response)
                        
                        # Check if 'moves' key exists and is a list
                        if 'moves' not in parsed_json:
                            print(f"{Fore.YELLOW}Warning: No 'moves' key in parsed response: {parsed_response[:100]}{Fore.RESET}")
                            self.error_steps += 1
                            self.need_new_plan = True
                            continue
                            
                        moves = parsed_json["moves"]
                        
                        # Validate moves are in the correct format
                        valid_moves = ["UP", "DOWN", "LEFT", "RIGHT"]
                        moves = [move for move in moves if move in valid_moves]
                        
                        if moves:
                            print(f"{Fore.GREEN}Got valid moves: {moves}{Fore.RESET}")
                            self.current_plan = moves
                            self.need_new_plan = False
                            self.empty_moves = 0
                            
                            # Store the moves for this round
                            if self.round_count > 0:
                                round_key = f"round_{self.round_count}"
                                if round_key in self.rounds_data:
                                    self.rounds_data[round_key]["moves"] = moves
                                
                        else:
                            print(f"{Fore.YELLOW}Empty or invalid moves list: {parsed_json.get('moves', [])}{Fore.RESET}")
                            self.empty_moves += 1
                            self.empty_steps += 1
                            
                            # Log reasoning if available
                            if 'reasoning' in parsed_json:
                                print(f"{Fore.CYAN}Reasoning: {parsed_json['reasoning']}{Fore.RESET}")
                                
                    except json.JSONDecodeError as e:
                        print(f"{Fore.RED}JSON decode error: {e}\nResponse: {parsed_response[:100]}...{Fore.RESET}")
                        self.error_steps += 1
                        self.need_new_plan = True
                    except KeyError as e:
                        print(f"{Fore.RED}Key error: {e}\nResponse: {parsed_response[:100]}...{Fore.RESET}")
                        self.error_steps += 1
                        self.need_new_plan = True
                        
                except Exception as e:
                    print(f"{Fore.RED}Error getting LLM response: {str(e)}{Fore.RESET}")
                    traceback.print_exc()
                    self.error_steps += 1
                    self.need_new_plan = True
            
            # Execute next move if we have one
            if self.current_plan:
                move = self.current_plan.pop(0)
                print(f"Executing move: {move}")
                
                # Keep track of the move
                self.current_round_moves.append(move)
                
                # Check if apple was eaten
                old_score = self.game.score
                
                # Execute the move
                self.game.move(move)
                
                # Check if an apple was eaten (score increased)
                if self.game.score > old_score:
                    # Record the current round moves
                    self.move_history.append(self.current_round_moves)
                    
                    # Reset for the next round
                    self.current_round_moves = []
                    self.round_count += 1
                    
                    # Store the new apple position
                    if self.game.apple:
                        self.apple_positions.append(self.game.apple)
                        
                        # Create a new round entry
                        round_key = f"round_{self.round_count}"
                        self.rounds_data[round_key] = {
                            "apple_position": self.game.apple,
                            "moves": [],
                            "primary_response_times": [],
                            "secondary_response_times": [],
                            "primary_token_stats": [],
                            "secondary_token_stats": []
                        }
                
                # Check if we need a new plan
                if not self.current_plan:
                    self.need_new_plan = True
            else:
                # No moves to execute, we need a new plan
                self.need_new_plan = True
            
            # Check for game over conditions
            if self.empty_moves >= self.max_empty_moves:
                print(f"{Fore.YELLOW}Game over: Maximum consecutive empty moves reached ({self.max_empty_moves}){Fore.RESET}")
                self.game.game_over = True
                self.game.collision_type = "max_empty_moves"
            
            # Update display
            if self.game.gui:
                self.game.gui.draw(self.game, self.round_count, self.game.steps)
                pygame.display.flip()
            
            # Control game speed
            time.sleep(self.move_delay)
        
        # Save the final round's moves if the game ended before eating an apple
        if self.current_round_moves:
            self.move_history.append(self.current_round_moves)

    def set_log_dir(self, log_dir):
        """Set the log directory.
        
        Args:
            log_dir: Path to the log directory
        """
        self.log_dir = log_dir
        
    def get_statistics(self):
        """Get game statistics.
        
        Returns:
            Dictionary containing game statistics
        """
        # Calculate response time statistics
        primary_response_stats = {
            "avg_primary_response_time": sum(self.primary_response_times) / len(self.primary_response_times) if self.primary_response_times else 0,
            "min_primary_response_time": min(self.primary_response_times) if self.primary_response_times else 0,
            "max_primary_response_time": max(self.primary_response_times) if self.primary_response_times else 0
        }
        
        secondary_response_stats = {
            "avg_secondary_response_time": sum(self.secondary_response_times) / len(self.secondary_response_times) if self.secondary_response_times else 0,
            "min_secondary_response_time": min(self.secondary_response_times) if self.secondary_response_times else 0,
            "max_secondary_response_time": max(self.secondary_response_times) if self.secondary_response_times else 0
        }
        
        # Get token statistics
        primary_token_stats = self.llm_client.get_token_stats() if self.llm_client else {}
        secondary_token_stats = self.parser_client.get_token_stats() if self.parser_client else {}
        
        # Combine all statistics
        statistics = {
            "primary_response_stats": primary_response_stats,
            "secondary_response_stats": secondary_response_stats,
            "primary_token_stats": primary_token_stats,
            "secondary_token_stats": secondary_token_stats,
            "apple_positions": self.apple_positions,
            "move_history": self.move_history,
            "rounds_data": self.rounds_data,
            "round_count": self.round_count,
            "empty_steps": self.empty_steps,
            "error_steps": self.error_steps,
            "parser_usage_count": self.parser_usage_count
        }
        
        return statistics 