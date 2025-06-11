seems the --max-steps option is not being respected.

Maybe it's the step calculation that's wrong.

the --max-games is not being respected, neither.




I just removed # Skip duplicate move entries
if self.moves and self.moves[-1] == move:
    return   in game_data.py . Also, self.game_state.reset() is now back in geme_controller.py 


The step count is not right.

