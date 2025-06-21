# Quick-and-Dirty Roadmap — LLM Plays Snake (v0, vague ideas)

> **This file is intentionally rough.**  We care more about seeing something work than about polished code or perfect architecture at this stage.

## The Vague Idea

What if... we just made a snake game and asked an LLM to play it? Like, super simple. No fancy architecture, no OOP, just functions and global variables. Maybe it works, maybe it doesn't. Let's find out!

## What We Want (Rough Goals)

1. **Basic Snake Game**: Snake moves, eats apples, grows, dies if hits wall/self
2. **LLM Brain**: Ask some LLM API "hey, what direction should the snake go?"
3. **Console Output**: Just print stuff to terminal, no fancy logging files - yet.
4. **Simple Prompt**: Short and sweet - tell LLM the game state, ask for moves to be made
5. **Non-OOP**: Functions, global vars, keep it simple stupid (KISS)

## Technical Approach (Very Rough)

- `main.py` - main game loop, handles pygame events
- `snake_game.py` - pygame drawing + game logic
- `llm_client.py` - simple functions to call LLM APIs, no classes

## Success Criteria

If the snake moves around and doesn't immediately crash, and the LLM sometimes makes reasonable moves, we'll call it a win. 

