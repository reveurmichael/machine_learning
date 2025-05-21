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
SNAKE_HEAD_C = (255, 140, 0)  # Bright orange for the head


# GUI
PIXEL = 100
WIDTH_HEIGHT = 800
APP_WIDTH = 800
APP_HEIGHT = 600
TIME_DELAY = 40
TIME_TICK = 280

GRID_SIZE = 10

DIRECTIONS = {
    "UP": (0, 1),  # No change in x, increase y (move up)
    "RIGHT": (1, 0),  # Increase x, no change in y (move right)
    "DOWN": (0, -1),  # No change in x, decrease y (move down)
    "LEFT": (-1, 0),  # Decrease x, no change in y (move left)
}

PROMPT_TEMPLATE_TEXT = """
You are an AI agent controlling a snake in the classic Snake game on a 10×10 grid. Coordinates range from (0,0) at the **bottom-left** to (9,9) at the **top-right**.

You are given the following inputs:
  • Head position: TEXT_TO_BE_REPLACED_HEAD_POS — (x, y) of the snake's head.
  • Current direction: TEXT_TO_BE_REPLACED_CURRENT_DIRECTION — One of "UP", "DOWN", "LEFT", "RIGHT", or "NONE".
  • Body cells: TEXT_TO_BE_REPLACED_BODY_CELLS — List of (x, y) positions occupied by the snake's body (excluding the head).
  • Apple position: TEXT_TO_BE_REPLACED_APPLE_POS — (x, y) coordinate of the apple.

## MOVEMENT RULES:
1. Valid directions: "UP", "DOWN", "LEFT", "RIGHT".
2. Direction vectors: 
   { "UP": (0, 1), "DOWN": (0, -1), "LEFT": (-1, 0), "RIGHT": (1, 0) }
3. The snake **cannot** reverse direction (e.g., if going UP, cannot move DOWN next). If the direction is "NONE", any move is allowed.
4. The snake **dies** if it:
   - Collides with its body (which shifts every step),
   - Moves outside the grid bounds (0 ≤ x < 10, 0 ≤ y < 10).
5. Eating the apple at TEXT_TO_BE_REPLACED_APPLE_POS:
   - Increases score by 1.
   - Grows the body by 1 segment (tail doesn't shrink on next move).

## COORDINATE SYSTEM:
- UP = y + 1
- DOWN = y - 1
- RIGHT = x + 1
- LEFT = x - 1

Example Moves from (1,1):
  • UP → (1,2)
  • DOWN → (1,0)
  • RIGHT → (2,1)
  • LEFT → (0,1)

## OBJECTIVE:
Plan a safe path for the head to reach the apple, avoiding collisions. Output a JSON object whose "moves" field is a sequence of moves that leads the head now at TEXT_TO_BE_REPLACED_HEAD_POS to eat the apple at TEXT_TO_BE_REPLACED_APPLE_POS.  

## REQUIRED OUTPUT FORMAT:
Return a **JSON object** in this **exact format**:

{
  "moves": ["MOVE1", "MOVE2", ...],
  "reasoning": "..." 
}

* "moves" must be a list of valid directions unless the apple is reachable in fewer.
* "reasoning" must be a brief explanation (1–2 sentences) of the path-planning rationale.
* If **no safe path** moves exists, return:

{ "moves": [], "reasoning": "NO_PATH_FOUND" }


## CONSTRAINTS:

* Must not reverse direction on the first move.
* Avoid collisions with walls and body.
* After eating the apple, avoid traps—ensure there's at least one legal move afterward.
* Use Manhattan distance as a heuristic, but prioritize safety.
* Snake movement update (per move):

  * New head = head + direction
  * Head becomes body segment
  * Tail is removed **unless** apple is eaten (in that case, tail remains)

## FINAL POSITION CHECK:

Make sure the total number of LEFT/RIGHT and UP/DOWN moves matches the apple's offset:

* From TEXT_TO_BE_REPLACED_HEAD_POS to TEXT_TO_BE_REPLACED_APPLE_POS:

  * x1 ≤ x2 → RIGHTs - LEFTs = x2 - x1
  * x1 > x2 → LEFTs - RIGHTs = x1 - x2
  * y1 ≤ y2 → UPs - DOWNs = y2 - y1
  * y1 > y2 → DOWNs - UPs = y1 - y2
    → In your case: TEXT_TO_BE_REPLACED_ON_THE_TOPIC_OF_MOVES_DIFFERENCE

## EDGE CASES TO CONSIDER:

* If apple is behind current direction, detour safely.
* If snake has no body, go directly toward apple.
* Avoid hugging walls unnecessarily, especially near apple.

Now, analyze the game state and return the JSON output. Return only a valid JSON object in the exact format:
{
  "moves": ["MOVE1", "MOVE2", ...],
  "reasoning": "..." 
}

"""

# Parser prompt for the second LLM
PARSER_PROMPT_TEMPLATE = """Extract a valid JSON object from this Snake game LLM response:

```
{response}
```

FORMAT REQUIREMENTS:
{
  "moves": ["MOVE1", "MOVE2", ...],
  "reasoning": "brief explanation"
}

RULES:
- "moves" must be a list of directions from: "UP", "DOWN", "LEFT", "RIGHT"
- Only include valid moves to reach the apple
- "reasoning" must be a brief explanation for the chosen path
- If no valid path exists or can't be determined, use: { "moves": [], "reasoning": "NO_PATH_FOUND" }

Return ONLY the JSON object without any additional text.
"""
