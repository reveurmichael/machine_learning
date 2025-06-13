No, having “SOMETHING_IS_WRONG” absent from `planned_moves` is by design and does not break any of the accounting / validation logic:

1. What the checks use  
   • Game-level or round-level **executed-moves** arrays (`detailed_history.moves` and `rounds_data.*.moves`) are the source of truth for:
     – `steps` count in the game root.  
     – `empty_steps`, `something_is_wrong_steps`, `invalid_reversals`, etc.  
     – JSON-sanity function `validate_game_summary()` (it compares those executed-moves lists with `steps`).  
   • `planned_moves` is treated as auxiliary metadata: “what the LLM said it intended to do when it first answered.”  Nothing in the stats logic or validator sums or cross-checks that list.

2. Why the sentinel should stay out of `planned_moves`  
   • `planned_moves` reflects the LLM’s original plan, which naturally never contains sentinel strings—the sentinels are injected later by the engine to mark runtime events (empty tick, invalid reversal, explicit error).  
   • Mixing them into `planned_moves` would corrupt any downstream analysis of LLM quality (e.g., “how often did the LLM plan an immediate reversal?”).

3. Impact on counts / assertions  
   • All step counts and integrity assertions rely only on executed-moves lists, where the sentinel **is** present.  
   • Therefore every “+=1” to `steps` has a matching item in `detailed_history.moves`, keeping invariants intact.  
   • Planned-moves length has no bearing on those invariants, so leaving the sentinel out cannot cause a mismatch.

4. Replay safety  
   • Replayer ignores `planned_moves` and drives the game from the executed-moves array, where the sentinel is present and handled (we added it to the non-moving set).

So no additional change is required; the data model remains internally consistent while keeping `planned_moves` semantically clean.