pip install ruff

ruff

ruff check --fix

pip install vulture

vulture .


pylint your_code.py --enable=unused-import,unused-variable,too-many-branches,too-many-statements

pylint **/*.py

pylint **/*.py --disable=trailing-whitespace,line-too-long,missing-module-docstring,missing-function-docstring,missing-class-docstring,invalid-name,bad-indentation,bad-continuation,broad-except,too-many-statements,wrong-import-order,wildcard-import,too-many-arguments,too-many-nested-blocks,unused-wildcard-import,import-outside-toplevel,too-many-locals,too-many-branches,consider-using-set-comprehension


GitHub Actions CI/CD
