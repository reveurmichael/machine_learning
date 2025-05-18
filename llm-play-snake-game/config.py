"""
Configuration settings for the Snake game.
Contains game parameters, colors, and prompt templates.
"""

# COLORS
SNAKE_C = (209, 204, 192)
APPLE_C = (192, 57, 43)
BG = (44, 44, 84)  # background
APP_BG = (240, 240, 240)
GRID_BG = (44, 44, 84)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREY = (113, 128, 147)
GREY2 = (189, 195, 199)
GREY3 = (87, 96, 111)

# GUI
PIXEL = 100
WIDTH_HEIGHT = 800
APP_WIDTH = 800
APP_HEIGHT = 600
TRAIN_HEIGHT = 500
TIME_DELAY = 40
TIME_TICK = 280

# Game
ROW = 10
DIRECTIONS = {
    "UP": (0, -1),
    "RIGHT": (1, 0),
    "DOWN": (0, 1),
    "LEFT": (-1, 0)
}

# LLM Prompt Template
PROMPT_TEMPLATE = """
You are an AI controlling a snake in a classic Snake game. Your goal is to provide MULTIPLE MOVES at once to help the snake eat apples and survive.

Game State:
{game_state}

Current Score: {score}
Steps Taken: {steps}

GAME OBJECTIVE:
Your goal is to eat as many apples (A) as possible without dying. Each apple increases your score by 1 point.

GAME MECHANICS:
1. The snake dies if it hits a wall (| or -) or its own body (S)
2. The snake's head (H) shows which direction you're currently moving
3. When you eat an apple, the snake grows longer, making it harder to avoid collisions
4. The game board is enclosed by walls - you cannot move past the edges

HOW TO PROVIDE MULTIPLE STEPS:
1. Analyze the current state and plan a safe path toward the apple
2. Provide a SEQUENCE of 5-10 moves that will help the snake reach the apple
3. Format your response as a numbered list of directions, for example:
   1. UP
   2. RIGHT
   3. RIGHT
   4. UP
   5. LEFT

RULES FOR YOUR RESPONSE:
- You MUST provide at least 5 sequential moves (unless you expect to reach the apple sooner)
- Valid directions are: UP, DOWN, LEFT, RIGHT
- You cannot move in the opposite direction of your current movement
- If you expect to reach the apple within your sequence, continue planning moves after eating it

THINK STEP BY STEP about the best sequence of moves, then output your final decision.
""" 