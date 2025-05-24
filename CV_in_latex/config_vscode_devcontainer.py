import os
import json

def create_devcontainer_config():
    """
    Creates the .devcontainer/devcontainer.json file with LaTeX environment configuration
    """
    # Create .devcontainer directory if it doesn't exist
    if not os.path.exists('.devcontainer'):
        os.makedirs('.devcontainer')
    
    # DevContainer configuration
    devcontainer_config = {
        "name": "LaTeX CV Environment",
        "dockerComposeFile": ["../docker-compose.yml"],
        "service": "cv-builder",
        "workspaceFolder": "/latex",
        "customizations": {
            "vscode": {
                "extensions": ["james-yu.latex-workshop"],
                "settings": {
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
                    "latex-workshop.view.pdf.viewer": "tab"
                }
            }
        },
        "remoteUser": "root"
    }
    
    # Write configuration to file
    with open('.devcontainer/devcontainer.json', 'w') as f:
        json.dump(devcontainer_config, f, indent=4)
    
    print(f".devcontainer/devcontainer.json created successfully.")

if __name__ == "__main__":
    create_devcontainer_config() 