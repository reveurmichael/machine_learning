 Unless for really commmon utils, no need to put things into the ./extensions/common/ folder or similar folders because I want things to be standalone, each and every of the new extensions. But we can regard the ./extensions/common/ folder as a folder for REALLY common utils, that no one will forget about its presence, then, an extension, plus the common folder, that is regarded as standalone as well.

 But we should not be sharing code between extensions. It's forbidden.


 
blabla-v0.01 + common is standalone.

blabla-v0.02 + common is standalone.

blabla-v0.03 + common is standalone.

blabla-v0.04 + common is standalone. (though, only heuristics will have v0.04).


