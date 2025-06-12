I really love the idea of having INVALID_REVERSAL in the game_N.json file because it makes things to be so obvious. 



Because those two “synthetic” symbols let you spot every control-flow hiccup at a glance.

1. INVALID_REVERSAL  
   • Appears only when the LLM proposes a move that would make the snake reverse into itself (e.g. last direction = RIGHT, proposal = LEFT).  
   • When you scan the `moves` array you can immediately see the exact step, the direction that was rejected, and how often it happens.  
   • That makes it trivial to answer questions like  
     – “Is the model learning the no-reverse rule?”  
     – “Does it break the rule more often under time pressure?”  
   • It also keeps the length of `moves` equal to `steps` without hiding the fact that a tick was consumed by a bad command.
