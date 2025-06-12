I really love the idea of having EMPTY in the game_N.json file because it makes things to be so obvious. 




Because each marker exposes a different “blind spot” in the model’s behaviour, so when you skim the JSON you can tell—without replaying the game—why progress stalled.

1. INVALID_REVERSAL  
   • Logged when the LLM tries to send the snake straight back onto itself (e.g. last
     direction = RIGHT, proposal = LEFT).  
   • Every such entry pins down the exact round and the exact step where the rule was
     broken.  
   • By just counting these markers you instantly see how often the model still
     violates the no-reverse constraint, or whether it improves after fine-tuning.

2. EMPTY  
   • Logged when the LLM’s reply for a whole round contains an **empty `moves` list** –
     i.e. the model answered “[]”.  
   • Highlights rounds where the agent could not think of any legal move at all.  
   • Lets you separate “no idea” failures from parsing errors or time-outs, because
     the round is otherwise perfectly well-formed, just devoid of content.

Together they turn the `moves` timeline into a self-explanatory audit trail:

• normal direction strings → “model had a plan and we executed it”  
• INVALID_REVERSAL   → “model broke the game’s basic rule here”  
• EMPTY       → “model could not come up with a move at this round”

That one-line classification is enough to build quick stats, spot pathological
games, or jump straight to the rounds that need debugging—all without crawling
through raw prompts and responses.