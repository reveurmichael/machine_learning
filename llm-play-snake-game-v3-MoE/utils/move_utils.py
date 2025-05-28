"""
Utility module for movement calculations in the Snake game.
"""

def calculate_move_differences(head_pos, apple_pos):
    """Calculate the expected move differences based on head and apple positions.
    
    Args:
        head_pos: Position of the snake's head as [x, y]
        apple_pos: Position of the apple as [x, y]
        
    Returns:
        String describing the expected move differences with actual numbers
    """
    head_x, head_y = head_pos
    apple_x, apple_y = apple_pos
    
    # Calculate horizontal differences
    x_diff_text = ""
    if head_x <= apple_x:
        x_diff = apple_x - head_x
        x_diff_text = f"#RIGHT - #LEFT = {x_diff} (= {apple_x} - {head_x})"
    else:
        x_diff = head_x - apple_x
        x_diff_text = f"#LEFT - #RIGHT = {x_diff} (= {head_x} - {apple_x})"
    
    # Calculate vertical differences
    y_diff_text = ""
    if head_y <= apple_y:
        y_diff = apple_y - head_y
        y_diff_text = f"#UP - #DOWN = {y_diff} (= {apple_y} - {head_y})"
    else:
        y_diff = head_y - apple_y
        y_diff_text = f"#DOWN - #UP = {y_diff} (= {head_y} - {apple_y})"
    
    return f"{x_diff_text}, and {y_diff_text}"

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