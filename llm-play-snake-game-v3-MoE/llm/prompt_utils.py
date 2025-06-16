"""
Prompt generation system for language models.
Functions for creating and formatting prompts for primary and secondary LLMs 
in the Snake game architecture.
"""

from config import PROMPT_TEMPLATE_TEXT_PRIMARY_LLM, PROMPT_TEMPLATE_TEXT_SECONDARY_LLM
from utils.moves_utils import calculate_move_differences

def prepare_snake_prompt(head_position, body_positions, apple_position, current_direction):
    """Prepare a prompt for the primary LLM to determine the next snake move.
    
    Constructs a structured prompt with game state information including
    the snake's head position, body positions, apple position, and current direction.
    
    Args:
        head_position: [x, y] position of the snake's head
        body_positions: List of [x, y] positions for the snake's body
        apple_position: [x, y] position of the apple
        current_direction: Current direction of movement
        
    Returns:
        Formatted prompt string
    """
    # Get head position in (x, y) format for prompt
    head_x, head_y = head_position
    head_pos = f"({head_x},{head_y})"
    
    # Get current direction string
    direction_str = current_direction if current_direction else "NONE"
    
    # Format body cells
    body_cells_str = format_body_cells_str(body_positions)
    
    # Get apple position
    apple_x, apple_y = apple_position
    apple_pos = f"({apple_x},{apple_y})"
    
    # Calculate the expected move differences using the utility function
    move_differences = calculate_move_differences(head_position, apple_position)
    
    # Create a prompt from the template text using string replacements
    prompt = PROMPT_TEMPLATE_TEXT_PRIMARY_LLM
    prompt = prompt.replace("TEXT_TO_BE_REPLACED_HEAD_POS", head_pos)
    prompt = prompt.replace("TEXT_TO_BE_REPLACED_CURRENT_DIRECTION", direction_str)
    prompt = prompt.replace("TEXT_TO_BE_REPLACED_BODY_CELLS", body_cells_str)
    prompt = prompt.replace("TEXT_TO_BE_REPLACED_APPLE_POS", apple_pos)
    prompt = prompt.replace("TEXT_TO_BE_REPLACED_ON_THE_TOPIC_OF_MOVES_DIFFERENCE", move_differences)
    
    return prompt

def create_parser_prompt(llm_response, head_pos=None, apple_pos=None, body_cells=None):
    """Create a prompt for the secondary LLM to parse the output of the primary LLM.
    
    Takes the raw response from the primary LLM and creates a prompt for the 
    secondary LLM to extract structured move data.
    
    Args:
        llm_response: The raw response from the primary LLM
        head_pos: Optional head position string in format "(x, y)"
        apple_pos: Optional apple position string in format "(x, y)"
        body_cells: Optional body cells string in format "[(x1, y1), (x2, y2), ...]"
        
    Returns:
        Prompt for the secondary LLM
    """
    # Use string replacement for the prompt template
    parser_prompt = PROMPT_TEMPLATE_TEXT_SECONDARY_LLM.replace("TEXT_TO_BE_REPLACED_FIRST_LLM_RESPONSE", llm_response)

    # Replace head and apple position placeholders if provided
    if head_pos:
        parser_prompt = parser_prompt.replace("TEXT_TO_BE_REPLACED_HEAD_POS", head_pos)
    if apple_pos:
        parser_prompt = parser_prompt.replace("TEXT_TO_BE_REPLACED_APPLE_POS", apple_pos)
    if body_cells:
        parser_prompt = parser_prompt.replace("TEXT_TO_BE_REPLACED_BODY_CELLS", body_cells)

    return parser_prompt

def format_body_cells_str(body_positions):
    """Format the snake body cells as a string representation.

    Args:
        body_positions: List of [x, y] coordinates of the snake segments

    Returns:
        String representation of body cells in format: "[(x1,y1), (x2,y2), ...]"
    """
    body_cells = []

    # Format each position as a tuple string
    for x, y in body_positions:
        body_cells.append(f"({x},{y})")

    return "[" + ", ".join(body_cells) + "]"

