
In every `game_N.json` there are two different move lists, and they come from two separate moments in the game loop:

1. planned_moves  
   • When the primary LLM answers the prompt “Here is the board, give me your next plan”, it usually returns something like  
     `["DOWN", "DOWN", "LEFT", "LEFT", "UP"]`.  
   • That entire array is stored immediately in `record_planned_moves()` and written into  
     `round_k["planned_moves"]`.  
   • It is a **static snapshot of the LLM’s intention for that round**.  
   • It is never altered afterwards; if the snake eats an apple early or hits a wall, the list is
     kept intact so you can later inspect “What did the model *want* to do?”.

2. moves  
   • Every time the game actually executes a single step, `record_move()` appends that step to  
     `round_k["moves"]` (and to the session-level `moves` list).  
   • Therefore this list is the **ground-truth history of what really happened on the board**.  
   • Its length equals the game’s `steps` counter.  
   • It can be shorter, longer, or completely different from `planned_moves`, for example:  
     – If the apple is reached earlier than planned, execution of the plan stops and a new
       LLM request is made (remaining planned steps are never executed, so they never appear
       in `moves`).  
     – If the plan contained an illegal reversal, the engine skips that entry; `planned_moves`
       still shows it, but `moves` doesn’t.

Quick mnemonic  
• planned_moves → “model’s proposal for the upcoming round”.  
• moves → “what the snake actually did, one step at a time”.

Having both lets you analyse planning quality versus real-world outcome: you can detect
over-planning, illegal moves, or how often the snake deviates from its own script.