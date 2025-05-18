"""
Minimal configuration settings for the Snake game.
Contains game parameters and colors without problematic f-strings.
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
TIME_DELAY = 40
TIME_TICK = 280

# Game
SNAKE_SIZE = 10
DIRECTIONS = {
    "UP": (0, 1),
    "RIGHT": (1, 0),
    "DOWN": (0, -1),
    "LEFT": (-1, 0)
}

# The PROMPT_TEMPLATE text without f-string evaluation
PROMPT_TEMPLATE_TEXT = """
You are an AI agent that controls a snake in a classic Snake game, with coordinates (r,c) from (0,0) at the bottom-left to (9, 9) at the top-right. You will be given:

INPUT VARIABLES:
  • HEAD_POS (r,c)       – Snake-head coordinates.
  • DIRECTION    – One of "UP," "DOWN," "LEFT," "RIGHT," or "NONE."
  • BODY_CELLS           – List of (r,c) positions occupied by the snake body (excluding the head).
  • APPLE_POS (r,c)      – Apple coordinates.

MOVEMENT RULES:
  1. Valid moves are "UP", "DOWN", "LEFT", "RIGHT". 
  2. DIRECTIONS = { "UP": (0, 1), "RIGHT": (1, 0), "DOWN": (0, -1), "LEFT": (-1, 0) }
  3. You cannot move in the exact opposite of DIRECTION on your first move.  
  4. The snake dies if head moves into any BODY_CELLS or outside 0 ≤ r < SIZE, 0 ≤ c < SIZE.  
  5. Eating the apple at APPLE_POS increments score by 1 and grows the body by 1 segment.

COORDINATE SYSTEM (IMPORTANT):
  The origin (0,0) is at the BOTTOM-LEFT of the grid.
  • UP: Increases row coordinate (moves toward row SIZE-1)
  • DOWN: Decreases row coordinate (moves toward row 0)
  • RIGHT: Increases column coordinate (moves toward column SIZE-1)
  • LEFT: Decreases column coordinate (moves toward column 0)

EXAMPLE GRID (4×4) WITH COORDINATES:
  (3,0)  (3,1)  (3,2)  (3,3)
  (2,0)  (2,1)  (2,2)  (2,3)
  (1,0)  (1,1)  (1,2)  (1,3)
  (0,0)  (0,1)  (0,2)  (0,3)

EXAMPLE MOVES FROM POSITION (1,1):
  • UP → (2,1)
  • DOWN → (0,1)
  • RIGHT → (1,2)
  • LEFT → (1,0)

OBJECTIVE:
  Plan a safe path to reach the apple without colliding. Output a sequence of moves (length 5–20) that leads the head to APPLE_POS if possible.

OUTPUT FORMAT:
  1. (Optional) Briefly (1–2 sentences) explain your path-planning rationale.  
  2. Then provide a JSON object exactly in this form:
     {
       "moves": [ "MOVE1", "MOVE2", … ],
       "reasoning": "…"
     }
     – "moves" must be a list of 5 to 20 directions (each one of "UP," "DOWN," "LEFT," "RIGHT").  
     – If no safe path of ≤20 moves exists, respond exactly with:
       { "moves": [], "reasoning": "NO_PATH_FOUND" }

CONSTRAINTS:
  - Must output at least 5 moves unless you will reach the apple in fewer.  
  - Do not reverse direction on your first move.  
  - Avoid collisions with walls or the body.  
  - When planning, assume your body will grow by 1 after eating the apple. Do not plan a move sequence that, after apple consumption, leaves you with no legal exit on the next turn if avoidable.
  - Use Manhattan distance to guide you toward the apple, but ALWAYS avoid any move that would collide with a wall or your own body. If you must detour around your own tail, do so.

EDGE CASES:
  - If apple is behind you but the only path is to circle around, you must output that circle.  
  - If the snake currently has no body (just head), any move is fine as long as it's toward the apple.  
  - If the apple is adjacent to the wall, be extra careful not to hug the wall.

Now, analyze the provided state and output your final answer.
""" 