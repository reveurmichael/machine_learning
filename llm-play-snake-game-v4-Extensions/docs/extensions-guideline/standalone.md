## VITAL: standalone should be very visible, across all extensions, but common folder is important

For somewhat commmon utils, put things into the ./extensions/common/ folder. We can regard the ./extensions/common/ folder as a folder for somewhat common utils (common for this moment, or maybe will be used in the future), that no one will forget about its presence, then, an extension blabla-v0.0N, plus the common folder, those two together will be regarded as standalone as well. But we should not be sharing code between extensions. It's forbidden. blabla-v0.01 + common is standalone. blabla-v0.02 + common is standalone. blabla-v0.03 + common is standalone. blabla-v0.04 + common is standalone. (though, only heuristics will have v0.04; for other extensions, there is only v0.01, v0.02 and v0.03).

The common folder is important because, after all, each extension blabla-v0.0N, represents important conceptual ideas (e.g. heuristics, supervised learning, RL, etc.), and it's those conceptual ideas that should be highlighed in each extension folder blabla-v0.0N. Moving non-essential code into the common folder helps those conceptual ideas to be more visible.

