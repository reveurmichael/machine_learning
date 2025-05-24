import os
import json
import platform

def create_vscode_settings():
    """
    Creates the .vscode/settings.json file with LaTeX Workshop configuration
    """
    # Create .vscode directory if it doesn't exist
    if not os.path.exists('.vscode'):
        os.makedirs('.vscode')
    
    # LaTeX Workshop settings
    settings = {
        "latex-workshop.latex.tools": [
            {
                "name": "pdflatex",
                "command": "pdflatex",
                "args": [
                    "-synctex=1",
                    "-interaction=nonstopmode",
                    "-file-line-error",
                    "%DOC%"
                ]
            }
        ],
        "latex-workshop.latex.recipes": [
            {
                "name": "pdflatex",
                "tools": [
                    "pdflatex"
                ]
            }
        ],
        "latex-workshop.latex.autoBuild.run": "onSave",
        "latex-workshop.view.pdf.viewer": "tab",
        "latex-workshop.latex.magic.args": [
            "-synctex=1",
            "-interaction=nonstopmode",
            "-file-line-error",
            "%DOC%"
        ],
        "latex-workshop.message.error.show": True,
        "latex-workshop.message.warning.show": True
    }
    
    # Add platform-specific settings if needed
    system = platform.system()
    if system == "Windows":
        print("Adding Windows-specific settings...")
        # Windows-specific settings could be added here if needed
    
    # Write settings to file
    with open('.vscode/settings.json', 'w') as f:
        json.dump(settings, f, indent=4)
    
    print(f".vscode/settings.json created successfully.")
    print("Note: This directory is gitignored, so these settings won't be tracked by git.")

if __name__ == "__main__":
    create_vscode_settings() 