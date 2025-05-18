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

# Direction vectors as (dx, dy) where:
# x increases to the right, y increases upward
# This matches (0,0) at bottom-left as in the prompt
DIRECTIONS = {
    "UP": (0, 1),      # No change in x, increase y (move up)
    "RIGHT": (1, 0),   # Increase x, no change in y (move right)
    "DOWN": (0, -1),   # No change in x, decrease y (move down)
    "LEFT": (-1, 0)    # Decrease x, no change in y (move left)
}

# The PROMPT_TEMPLATE text without f-string evaluation
PROMPT_TEMPLATE_TEXT = """
You are an AI agent that controls a snake in a classic Snake game, with coordinates (x,y) from (0,0) at the bottom-left to (9, 9) at the top-right. You will be given:

INPUT VARIABLES:
  • HEAD_POS (x,y)       – Snake-head coordinates.
  • DIRECTION    – One of "UP," "DOWN," "LEFT," "RIGHT," or "NONE."
  • BODY_CELLS           – List of (x,y) positions occupied by the snake body (excluding the head).
  • APPLE_POS (x,y)      – Apple coordinates.

MOVEMENT RULES:
  1. Valid moves are "UP", "DOWN", "LEFT", "RIGHT". 
  2. DIRECTIONS = { "UP": (0, 1), "RIGHT": (1, 0), "DOWN": (0, -1), "LEFT": (-1, 0) }
  3. You cannot move in the exact opposite of DIRECTION on your first move.  
  4. The snake dies if head moves into any BODY_CELLS or outside 0 ≤ x < 10, 0 ≤ y < 10.  
  5. Eating the apple at APPLE_POS increments score by 1 and grows the body by 1 segment.

COORDINATE SYSTEM (IMPORTANT):
  The origin (0,0) is at the BOTTOM-LEFT of the grid.
  • UP: Increases y coordinate (moves toward y 9)
  • DOWN: Decreases y coordinate (moves toward y 0)
  • RIGHT: Increases x coordinate (moves toward x 9)
  • LEFT: Decreases x coordinate (moves toward x 0)
  • The first element of the tuple is the LEFT and RIGHT direction (x axis), the second element is the UP and DOWN direction (y axis).

EXAMPLE GRID (4×4) WITH COORDINATES:
  (0,3)  (1,3)  (2,3)  (3,3)
  (0,2)  (1,2)  (2,2)  (3,2)
  (0,1)  (1,1)  (2,1)  (3,1)
  (0,0)  (1,0)  (2,0)  (3,0)

EXAMPLE MOVES FROM POSITION (1,1):
  • UP → (1,2)

EXAMPLE MOVES FROM POSITION (1,1):
  • DOWN → (1,0)

EXAMPLE MOVES FROM POSITION (1,1):
  • RIGHT → (2,1)

EXAMPLE MOVES FROM POSITION (1,1):
  • LEFT → (0,1)

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

IMPORTANT:
  - Keep in mind that the coordinate system is (x,y) where x is the horizontal axis (left to right) and y is the vertical axis (bottom to top). Just like in middle school and high school math.

Now, analyze the provided state and output your final answer.
"""
