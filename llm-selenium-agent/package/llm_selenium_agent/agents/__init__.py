import os
import importlib
import glob
import inspect

current_directory = os.path.dirname(__file__)
python_files = glob.glob(os.path.join(current_directory, "*.py"))

# Define variables that might not be set if no files are found
module_name = None
module = None
class_name = None
class_obj = None

# Dynamically import all classes except __init__.py
for file_path in python_files:
    if os.path.basename(file_path) != "__init__.py":
        module_name = os.path.splitext(os.path.basename(file_path))[0]
        module = importlib.import_module(f".{module_name}", package=__package__)

        # Extract and register classes defined in the module
        for class_name, class_obj in inspect.getmembers(module, inspect.isclass):
            if class_obj.__module__ == module.__name__:
                globals()[class_name] = class_obj

# Remove temporary variables from namespace
del (
    os,
    importlib,
    glob,
    inspect,
    current_directory,
    python_files,
    file_path,
    module_name,
    module,
    class_name,
    class_obj,
)
