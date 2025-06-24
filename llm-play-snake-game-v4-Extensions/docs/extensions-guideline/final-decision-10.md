## TASK_DISCRIPTION_GOODFILES

Those files in the ./docs/extensions-guideline/ folder are called GOODFILES: 
- final-decision-N.md (N=0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
- agents.md
- app.md
- datasets_folder.md
- dashboard.md
- documentation-as-first-class-citizen.md
- extensions_move_guidelines.md
- Grid-Size-Directory-Structure-Compliance-Report.md
elegance.md
extensions-v0.01.md
extensions-v0.02.md
extensions-v0.03.md
extensions-v0.04.md
coordinate-system.md
cwd-and-logs.md
csv_schema-1.md
csv_schema-2.md
ai-friendly.md
config.md
gymnasium.md
heuristics_as_foundation.md
heuristics_to_supervised_pipeline.md
html.md
limits-manager-impact-on-extensions.md
lora.md
models.md
mutilple-inheritance.md
mvc-impact-on-extensions.md
mvc.md
naming_conventions.md
no-gui.md
npz-paquet.md
onnx.md
standalone.md
round.md
single-source-of-truth.md
scripts.md
vision-language-model.md
project-structure-plan.md


Don't modify GOODFILES. They are the single source of truth. Now, according to GOODFILES, update other md files (all of them located in the ROOT/docs/extensions-guideline/ folder). Don't modify System-Prompt.md. System-Prompt.md is important as well, though, it's not the final decision, not as authoritative as the GOODFILES. What is written in other non-GOODFILES.md files and non-System-Prompt.md files are less important than the GOODFILES. When there is a conflict, the final decisions of GOODFILES is the single source of truth. For topics unrelated to GOODFILES, those non-GOODFILES.md files and non-System-Prompt.md files are accepted as well, though sometimes it will make things less coherent and less elegant. In such cases, you can either:

- Flag them in the md file, with a TODO comment about why things are not coherent. Users (I and my colleagues) can then recheck those md files for TODO comments and make my own decision.
- Or, if the solutions to adopt is just so obvious, you can just do it, no need to go for this TODO comment.

## Now, Do TASK_DISCRIPTION_GOODFILES

Now, do the TASK_DISCRIPTION_GOODFILES. For md files in the folder ./docs/extensions-guideline/ (excluding GOODFILES), don't produce a lot of code. Depending on the context, sometimes a little bit of code for the main idea is enough. Sometimes a bit more. On the contrary, the motivation, the design philosophy, the implications, the interconnection of the concept/classes/python files etc. text are more important for me. Each md should not be excessively long. If the ideas are clear enough, you can go for a relatively short md, targeting 300 lines; if there is a lot to discuss, you can go for a longer md, targeting 500 lines. Concise and elegance for most cases, long format for the more complex cases.

So you basically have to rewrite those md files, most likely make them shorter and more concise, each of them, to make them coherent and with the objective of making the whole project elegant.

## GO AHEAD AND NOW, DO IT.

