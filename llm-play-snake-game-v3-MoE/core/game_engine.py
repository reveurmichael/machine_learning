"""
Game engine module for the Snake game.
Handles the game loop and move execution.
"""

import os
import json
import time
import pygame
import datetime
from colorama import Fore
from llm_client import LLMClient, LLMOutputParser
from utils.log_utils import save_to_file


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

    def set_game(self, game):
        """Set the game instance.
        
        Args:
            game: SnakeGame instance
        """
        self.game = game
        self.need_new_plan = True
        self.current_plan = []
        self.empty_moves = 0

    def get_llm_response(self):
        """Get a response from the LLM.
        
        Returns:
            Tuple of (parsed_response, parser_prompt)
        """
        # Create the prompt
        prompt = self._create_prompt()
        
        # Save the prompt
        if self.prompts_dir:
            prompt_file = os.path.join(
                self.prompts_dir,
                f"prompt_{self.game.steps}.txt"
            )
            save_to_file(prompt_file, prompt)
        
        # Get response from primary LLM
        response = self.llm_client.generate_response(prompt)
        
        # Save the response
        if self.responses_dir:
            response_file = os.path.join(
                self.responses_dir,
                f"response_{self.game.steps}.txt"
            )
            save_to_file(response_file, response)
        
        # Parse the response
        parsed_response, parser_prompt = self.parser_client.parse_and_format(
            response,
            self.game.head,
            self.game.apple,
            self.game.body
        )
        
        # Save the parser prompt and response
        if self.prompts_dir:
            parser_prompt_file = os.path.join(
                self.prompts_dir,
                f"parser_prompt_{self.game.steps}.txt"
            )
            save_to_file(parser_prompt_file, parser_prompt)
            
        if self.responses_dir:
            parsed_response_file = os.path.join(
                self.responses_dir,
                f"parsed_response_{self.game.steps}.txt"
            )
            save_to_file(parsed_response_file, parsed_response)
        
        return parsed_response, parser_prompt

    def _create_prompt(self):
        """Create a prompt for the LLM.
        
        Returns:
            The prompt string
        """
        return f"""You are playing a game of Snake. The goal is to eat the apple and grow as long as possible without hitting the walls or yourself.

Current game state:
- Head position: {self.game.head}
- Apple position: {self.game.apple}
- Body cells: {self.game.body}
- Score: {self.game.score}
- Steps: {self.game.steps}

Please provide a sequence of moves to reach the apple. Valid moves are: UP, DOWN, LEFT, RIGHT.
Format your response as a JSON object with a "moves" array containing the sequence of moves.

Example response:
{{
    "moves": ["UP", "RIGHT", "DOWN", "LEFT"]
}}
"""

    def run_game(self):
        """Run the game loop."""
        # Set up logging directories
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
                    parsed_response, _ = self.get_llm_response()
                    self.parser_usage_count += 1
                    
                    # Parse the moves
                    try:
                        moves = json.loads(parsed_response)["moves"]
                        if moves:
                            self.current_plan = moves
                            self.need_new_plan = False
                            self.empty_moves = 0
                        else:
                            self.empty_moves += 1
                            self.empty_steps += 1
                    except (json.JSONDecodeError, KeyError):
                        self.error_steps += 1
                        self.need_new_plan = True
                        
                except Exception as e:
                    print(f"{Fore.RED}Error getting LLM response: {str(e)}{Fore.RESET}")
                    self.error_steps += 1
                    self.need_new_plan = True
            
            # Execute next move
            if self.current_plan:
                move = self.current_plan.pop(0)
                self.game.move(move)
                
                # Check if we need a new plan
                if not self.current_plan:
                    self.need_new_plan = True
            
            # Check for game over conditions
            if self.empty_moves >= self.max_empty_moves:
                self.game.game_over = True
                self.game.collision_type = "max_empty_moves"
            
            # Update display
            if self.game.gui:
                self.game.gui.draw()
                pygame.display.flip()
            
            # Control game speed
            time.sleep(self.move_delay) 