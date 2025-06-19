## Question 

I am asking you a question. After each game finish, the system produces a game_N.json file. Will it update the summary.json as well? For example, at the beginning of the main.py mode, we set --max-games to 5. Then, I ctrl C to stop the session after 2 games and a bit into the 3rd game. At this point (let's say POINT_A), 

- will the summary.json be updated according to game 2 or game 3? 
- Then, at POINT_A, if I continue the game, with --max-games 5, will the system continue the game from game 3 till 5? 
- Or, at POINT_A, if I continue the game, with --max-games 10, will the system continue the game from game 3 till 10? or will it start from game 6 to game 10? 

## Answer

Here’s what happens in each situation:

1. Was summary.json updated with game 2 or game 3?
   • Only fully-finished games are recorded.  
   • `summary.json` is rewritten inside `process_game_over()` → `save_session_stats()` **after** a game ends cleanly.  
   • Because you stopped during game 3, that game never reached “game over”, so the last write is for game 2.  
   ⇒ At POINT_A the summary reflects the results of games 1 and 2 only.

2. Continue with `--max-games 5`
   • The continuation helper (`utils/continuation_utils.py`) reads `summary.json`, sees that `game_statistics.total_games == 2`, and sets the next game number to 3.  
   • With `--max-games 5` the new run will play games 3, 4 and 5.  
   ⇒ Exactly 3 more games are played so the session reaches 5 total.

3. Continue with `--max-games 10`
   • Again it resumes at game 3 (nothing changes about the next game number).  
   • But the overall limit is now 10, so it will play games 3 → 10.  
   • It will **not** skip ahead to 6; the numbering remains sequential.  
   ⇒ Eight additional games are played, bringing the session to 10 completed games.

Summary:
• `summary.json` always lags by one interrupted game.  
• Continuation always starts with the first un-finished game, and it respects the (possibly new) `--max-games` ceiling while keeping the original numbering.