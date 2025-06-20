## Planning

First, do it for v3 (files: /Users/lunde/Downloads/tmp3/mypy-v3-MoE).

Then, generate a list of types of all variables, functions, classes, etc. for the whole project. 

Then, go back to v4, forcing v4 to adopt the same types.

## Prompt/cursor rules:

Improve the code by adding precise and correct type hints, compatible with both `mypy` and `pyright`. Follow PEP 484 and PEP 561 standards strictly.

Your goals:
- Fix the most common or critical type issues first (e.g., missing annotations, incorrect argument types, incompatible return types).
- Do not change the functionality or refactor logic.
- Prefer specific types over generic ones (e.g., `list[int]` instead of `list`, `tuple[int, int]` over `Sequence[int]` if required).
- Use `Optional[...]` and `Union[...]` only when needed, based on usage.
- Add `# type: ignore` comments **only as a last resort** and explain why in a comment.
- If a 3rd-party import has no stubs, use `# type: ignore[import-untyped]` and optionally mention how to install the correct stub package.
- Be conservative and prioritize correctness over coverage.

Do not make structural or stylistic changes unrelated to typing.

IMPORTANT: Do it without any functionality change. 




