"""
Simple main loop for LLM Snake - keeps original UI but no OOP, no file logging

NOTE: This represents our "vague idea" turned into working code!
This main.py shows how you can quickly get a complex system running by keeping things simple.
We combine pygame, LLM integration, and game logic in one straightforward loop.

WHAT WE'VE ACCOMPLISHED IN V0:
- Working game loop with LLM integration
- Direct approach that gets results quickly
"""

import pygame
import time
from snake_game import (
    init_pygame, init_game, move_snake, draw_game, 
    get_game_state, get_score, get_steps, reset_game,
    parse_llm_response, get_next_planned_move
)
from llm_client import call_llm

# Short simple prompt - much shorter than original
# Starting with a concise prompt that covers the essentials
SHORT_PROMPT = """Snake game. Current state:

{game_state}

Score: {score} | Steps: {steps}

You are the snake head (H with arrow showing direction). Eat apples (A). Avoid walls and body (S).

Respond with multiple moves like: UP, RIGHT, RIGHT, DOWN, LEFT"""

def main():
    """
    Main game loop - keeps original structure but simplified.
    
    This function orchestrates the entire game experience:
    - Pygame initialization and setup
    - Event processing for user input
    - LLM communication and planning
    - Move execution and game state updates
    - Real-time display updates
    
    This demonstrates how a single function can coordinate
    multiple subsystems. In v1, we'll see how separating concerns into
    classes makes this more modular and testable.
    """
    print("🐍 Starting LLM Snake Game (Non-OOP Version)")
    
    # Initialize pygame and game state
    # Simple setup - gets us running quickly
    init_pygame()
    init_game()
    
    # Game settings - embedded configuration for quick prototyping
    # FUTURE IMPROVEMENT: In v1, these could be in a proper config system
    provider = "hunyuan"  # Change to "ollama" for local LLM
    clock = pygame.time.Clock()
    move_pause = 1.0  # Seconds between moves - good for observing LLM thinking
    game_active = True
    need_new_plan = True
    
    print(f"Using LLM provider: {provider}")
    print("Press ESC to quit, R to reset, SPACE to toggle speed")
    
    # Main game loop - coordinates all game systems
    running = True
    while running:
        # Event handling section
        # Direct event processing keeps things simple for v0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    # Reset game state
                    reset_game()
                    game_active = True
                    need_new_plan = True
                    print("🔄 Game reset!")
                elif event.key == pygame.K_SPACE:
                    # Toggle between fast and normal speed for different viewing experiences
                    move_pause = 0.1 if move_pause > 0.5 else 1.0
                    print(f"⚡ Speed: {'Fast' if move_pause < 0.5 else 'Normal'}")
        
        # Game logic section - the heart of our LLM integration
        if game_active:
            try:
                # LLM planning and move execution
                # This section shows how we can integrate AI decision-making
                # into a real-time game loop
                
                if need_new_plan:
                    # Generate new plan from LLM
                    # This is where the magic happens - AI controlling the game!
                    
                    # Get current game state for LLM analysis
                    state = get_game_state()
                    current_score = get_score()
                    current_steps = get_steps()
                    
                    # Create prompt with current game information
                    # Simple string formatting gets the job done
                    prompt = SHORT_PROMPT.format(
                        game_state=state,
                        score=current_score,
                        steps=current_steps
                    )
                    
                    print("\n" + "="*60)
                    print("🤖 Getting new plan from LLM...")
                    
                    # Call LLM and parse response
                    # Direct API call keeps the flow simple and understandable
                    llm_response = call_llm(prompt, provider)
                    next_move = parse_llm_response(llm_response)
                    
                    # Show what the LLM is thinking
                    print(f"LLM Response: {llm_response[:100]}...")
                    print(f"Next move: {next_move}")
                    
                    need_new_plan = False
                    
                    # Execute the first move if valid
                    if next_move:
                        game_active, apple_eaten = move_snake(next_move)
                        # If apple eaten, get new plan immediately for adaptive behavior
                        if apple_eaten:
                            print("🍎 Apple eaten! Getting new plan...")
                            need_new_plan = True
                    else:
                        print("⚠️ No valid move found")
                        
                else:
                    # Execute remaining planned moves
                    # This shows how we can batch AI decisions for efficiency
                    next_move = get_next_planned_move()
                    
                    if next_move:
                        print(f"🎯 Executing planned move: {next_move}")
                        game_active, apple_eaten = move_snake(next_move)
                        
                        # Adaptive replanning when environment changes
                        if apple_eaten:
                            print("🍎 Apple eaten! Getting new plan...")
                            need_new_plan = True
                    else:
                        # No more planned moves, time for new AI input
                        need_new_plan = True
                
                # Check for game over and provide feedback
                if not game_active:
                    print(f"\n💀 Game Over! Final Score: {get_score()}, Steps: {get_steps()}")
                    print("Press R to restart or ESC to quit")
                
                # Pause between moves for visualization
                # This timing allows users to see the LLM's thinking process
                time.sleep(move_pause)
                
            except Exception as e:
                # Graceful error handling keeps the demo running
                print(f"❌ Error in game loop: {e}")
                # Continue playing to maintain the demo experience
        
        # Always draw the game to maintain visual feedback
        # Continuous rendering keeps the UI responsive
        draw_game()
        
        # Control frame rate for smooth animation
        clock.tick(60)
    
    # Cleanup and farewell
    pygame.quit()
    print("Thanks for playing!")

if __name__ == "__main__":
    # Direct execution - simple and straightforward for v0
    main()
