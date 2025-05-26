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
# Pause time between sequential moves (in seconds)
MOVE_PAUSE = 1.0
# Pause time between games in replay mode (in seconds)
PAUSE_BETWEEN_GAMES_SECONDS = 1.0

# Game configuration
GRID_SIZE = 10
MAX_CONSECUTIVE_EMPTY_MOVES = 3  # Maximum consecutive empty moves before game over

DIRECTIONS = {
    "UP": (0, 1),  # No change in x, increase y (move up)
    "RIGHT": (1, 0),  # Increase x, no change in y (move right)
    "DOWN": (0, -1),  # No change in x, decrease y (move down)
    "LEFT": (-1, 0),  # Decrease x, no change in y (move left)
}


PROMPT_TEMPLATE_TEXT = """
You are an AI agent controlling a snake in the classic Snake game on a 10x10 grid. Coordinates range from (0,0) at the **bottom-left** to (9,9) at the **top-right**.

You are given the following inputs:
  • Head position: `TEXT_TO_BE_REPLACED_HEAD_POS` — (x,y) of the snake's head.  
  • Current direction: `"TEXT_TO_BE_REPLACED_CURRENT_DIRECTION"` — One of "UP", "DOWN", "LEFT", "RIGHT", or "NONE".  
  • Body cells: `TEXT_TO_BE_REPLACED_BODY_CELLS` — Ordered list of (x,y) positions occupied by the snake's body segments, from the segment immediately behind the head to the tail.  
  • Apple position: `TEXT_TO_BE_REPLACED_APPLE_POS` — (x,y) coordinate of the apple.

## SNAKE REPRESENTATION AND CONTINUITY:
1. The full snake S is `[HEAD, BODY1, BODY2, ..., BODYN]`.  
2. Each BODYi is adjacent (Manhattan distance = 1) to its predecessor (HEAD or BODYi-1).  
3. BODYN is the tail segment. The provided Body cells list (in the inputs) excludes the head and goes from BODY1 (nearest the head) to BODYN (the tail).  
4. Example valid S: `[(4,4) (head), (4,3), (4,2), (3,2), (3,1), (2,1), (2,2), (2,3), (3,3) (tail)]`; each consecutive pair differs by exactly 1 in either x or y.

## MOVEMENT UPDATE RULE:
Each move proceeds as follows:
1. **Compute new head**:  
   new_head = (head_x + dx[D], head_y + dy[D])
   where direction vectors are `{ "UP": (0, +1), "DOWN": (0, -1), "LEFT": (-1, 0), "RIGHT": (+1, 0) }`.  
2. **Check for collisions** (using the state before body shifts):  
  • If new_head is outside grid bounds (0 ≤ x < 10, 0 ≤ y < 10), the snake dies.  
  • If new_head coincides with any BODYi except the tail's current cell when that tail will move this turn, the snake dies.  
3. **Update body positions** (assuming no collision and within bounds):  
  a. Insert the old head position at the front of the BODY list (becomes BODY1).  
  b. If new_head == apple position, **grow**: do not remove the tail this turn (BODY length increases by 1).  
  c. Otherwise, **move normally**: remove the last element of BODY (the tail) so total length remains the same.  
4. **Set HEAD = new_head**; the BODY list is now updated from BODY1 to BODYN.

### Movement Example (Normal Move):
If S = `[HEAD (4,4), BODY1 (4,3)]` and you choose "LEFT":  
- new_head = (4-1, 4) = (3,4).  
- No collision if (3,4) is empty.  
- Insert old head (4,4) at front of BODY ⇒ BODY becomes `[(4,4), (4,3)]`.  
- Remove tail (4,3) (assuming not eating apple) ⇒ BODY becomes `[(4,4)]`.  
- Update HEAD to (3,4).  
- Resulting S = `[(3,4) (head), (4,4) (body1)]`.

### Movement Example (Eating Apple):
Suppose S = `[HEAD (5,5), BODY1 (5,4), BODY2 (5,3)]` and the apple is at (6,5). Current direction is "RIGHT".  
1. Choose "RIGHT": new_head = (5+1, 5) = (6,5).  
2. Check collisions: (6,5) is inside bounds and not occupied by BODY (BODY is at (5,4) and (5,3)), so safe.  
3. Update BODY:  
a. Insert old head (5,5) at front ⇒ BODY becomes `[(5,5), (5,4), (5,3)]`.  
b. Since new_head == apple position, **grow**: do NOT remove the tail. Now BODY = `[(5,5), (5,4), (5,3)]` plus the new head will be added next.  
4. Update HEAD = (6,5). The new full S = `[(6,5) (head), (5,5), (5,4), (5,3) (tail)]`. Body length increased by 1.

## MOVEMENT RULES:
1. Valid directions: "UP", "DOWN", "LEFT", "RIGHT".  
2. **Cannot reverse**: If current direction is "UP", you may not choose "DOWN" next; similarly for "LEFT"↔"RIGHT". If current direction is "NONE", any first move is allowed.  
3. **Death conditions** (the snake dies if):  
- HEAD moves outside grid bounds.  
- HEAD moves into a BODY cell that is not the tail's current cell when that tail is about to move.  
4. **Eating the apple** (when new_head == apple position):  
- Score increases by 1.  
- BODY grows by 1 (tail does not move on this step).  
- After eating, ensure there is at least one legal move on the next turn to avoid immediate trapping.

## COORDINATE SYSTEM:
- "UP" means y+1  
- "DOWN" means y-1  
- "RIGHT" means x+1  
- "LEFT" means x-1  

Example Moves from (1,1):  
• UP → (1,2)  
• DOWN → (1,0)  
• RIGHT → (2,1)  
• LEFT → (0,1)

## OBJECTIVE:
Plan a safe path for the HEAD to reach the APPLE, avoiding collisions and respecting all rules. Output a JSON object whose "moves" field is a sequence of directions that takes the head (now at `TEXT_TO_BE_REPLACED_HEAD_POS`) to eat the apple at `TEXT_TO_BE_REPLACED_APPLE_POS`.

## REQUIRED OUTPUT FORMAT:
Return a **JSON object** in this **exact format** (and nothing else!):

{
"moves": ["MOVE1", "MOVE2", …],
"reasoning": "…"
}

* "moves" must be a list of valid direction strings that lead from HEAD to APPLE safely.  
* "reasoning" must be a brief explanation (1-2 sentences) of your path-planning rationale.  
* If **no safe path** exists, return exactly:  
  { "moves": [], "reasoning": "NO_PATH_FOUND" }

## PATH-PLANNING GUIDELINES:
1. Must not reverse direction on the first move.  
2. Avoid collisions: walls, body segments, and future body positions given shifts.  
3. After eating the apple, there must be at least one legal move next turn (do not trap yourself).  
4. Use Manhattan distance as a heuristic, but prioritize safety over shortest path.  
5. Ensure the BODY remains continuous after each move (each adjacent pair differs by exactly 1 in x or y).

## FINAL POSITION CHECK:
Verify that the net number of horizontal and vertical moves in your proposed answer matches the apple's offset from the head:  
• Let (x1, y1) = `TEXT_TO_BE_REPLACED_HEAD_POS`, (x2, y2) = `TEXT_TO_BE_REPLACED_APPLE_POS`.  
 - If x1 ≤ x2 → #RIGHT - #LEFT = x2 - x1.  
 - If x1 > x2 → #LEFT - #RIGHT = x1 - x2.  
 - If y1 ≤ y2 → #UP - #DOWN = y2 - y1.  
 - If y1 > y2 → #DOWN - #UP = y1 - y2.  
→ In your case: TEXT_TO_BE_REPLACED_ON_THE_TOPIC_OF_MOVES_DIFFERENCE

## ADDITIONAL EDGE CASES TO CONSIDER:
* **Snake of Length 1**: If the snake has no body segments (BODY list is empty), you may move directly toward the apple unimpeded.  
* **Almost-Full Grid**: If the snake's body occupies most cells, ensure you leave a path open to avoid getting trapped—even if it's longer.  
* **Adjacent Apple with Blocking Body**: If the apple is adjacent but a body segment is between head and apple, plan a detour.  
* **Border Hugging**: Avoid hugging walls unnecessarily—especially near the apple—to prevent getting boxed in.  
* **Immediate Self-Collision**: Don't move into the cell the tail will vacate if the tail does **not** move (i.e., when eating an apple, the tail stays).

Now analyze the game state and return only the JSON output as specified. Do **not** include any additional text or code, do not return any python code or other code, do not return Latex—return only the valid JSON object in the exact format.
"""


# Parser prompt for the secondary LLM (formatting expert)
PARSER_PROMPT_TEMPLATE = """You are the secondary LLM in a Mixture-of-Experts system for a Snake game. Your job is to format the primary LLM's output into valid JSON.

Generate a valid JSON object from this Snake game primary LLM response (let's call it RESPONSE_1):

BEGINNING OF RESPONSE_1:

```
TEXT_TO_BE_REPLACED_FIRST_LLM_RESPONSE
```

END OF RESPONSE_1.

## WHAT RESPONSE_1 LOOKS LIKE:
RESPONSE_1 might look like this:
  <think> THINK_PROCESS_TEXT_OF_RESPONSE_1 </think>
  FINAL_OUTPUT_TEXT_OF_RESPONSE_1

## OUTPUT FORMAT REQUIREMENTS OF YOUR ANSWER:
{
  "moves": ["MOVE1", "MOVE2", ...],
  "reasoning": "brief explanation"
}


## COORDINATE SYSTEM TO HELP YOU UNDERSTAND RESPONSE_1:
- UP = y + 1
- DOWN = y - 1
- RIGHT = x + 1
- LEFT = x - 1

Example Moves from (1,1):
  • UP → (1,2)
  • DOWN → (1,0)
  • RIGHT → (2,1)
  • LEFT → (0,1)

## THE MAIN OBJECTIVE OF RESPONSE_1 SO THAT YOU CAN UNDERSTAND BETTER RESPONSE_1:

The main objective of RESPONSE_1 is to generate a list of moves that leads the snake, whose head is now at TEXT_TO_BE_REPLACED_HEAD_POS and whose body cells are in list format TEXT_TO_BE_REPLACED_BODY_CELLS, to eat the apple at TEXT_TO_BE_REPLACED_APPLE_POS.

## RULES:
- "moves" must be a list of directions from: "UP", "DOWN", "LEFT", "RIGHT"
- "reasoning" must be a brief explanation for the chosen path
- If after reading RESPONSE_1, according to your understanding, no valid path exists or can't be determined, or if you are not sure about the text's meaning of path planning, use: { "moves": [], "reasoning": "NO_PATH_FOUND" } or { "moves": [], "reasoning": "I_CANNOT_UNDERSTAND_THE_TEXT" }

## WHAT YOU SHOULD DO:
- First, if RESPONSE_1 starts with the word "ERROR" (capitalized), you should return { "moves": [], "reasoning": "ERROR" }.
- Second, if RESPONSE_1 contains the <think>...</think> tags, you should ignore THINK_PROCESS_TEXT_OF_RESPONSE_1 and rely solely on FINAL_OUTPUT_TEXT_OF_RESPONSE_1 to figure out your answer in JSON format.
- Third, if RESPONSE_1 does not contain the <think>...</think> tags, you should use use the whole text of RESPONSE_1 to figure out your answer in JSON format. Texts at the beginning of RESPONSE_1 are much less important than the texts torwards the end.

Return ONLY the JSON object without any additional text.
"""
