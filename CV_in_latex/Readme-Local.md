# Local LaTeX Build with VSCode/Cursor

This guide explains how to set up and build your LaTeX CV directly on your local machine using VSCode or Cursor.

## Advantages of Local Builds

- **Speed**: Direct compilation without container overhead
- **Simplicity**: Once set up, it's the fastest workflow
- **Integration**: Full editor features including preview, syntax highlighting, and autocomplete
- **Offline Use**: No internet connection required for builds

## Setup Process

### 1. Install LaTeX Workshop Extension

1. Open VSCode/Cursor
2. Go to Extensions (Ctrl+Shift+X or Cmd+Shift+X)
3. Search for "LaTeX Workshop"
4. Click "Install"

### 2. Install LaTeX Distribution

You need a LaTeX distribution installed on your system.

#### On macOS:

```bash
# Using Homebrew
brew install texlive
```

#### On Windows:
- Install [TeX Live](https://tug.org/texlive/windows.html)
- Ensure Perl is installed (required by some LaTeX packages)

#### On Linux:
```bash
# Ubuntu/Debian
sudo apt install texlive-full

# Fedora
sudo dnf install texlive-scheme-full
```

### 3. Configure VSCode

Use our configuration script to set up VSCode automatically:

```bash
python config_vscode_local.py
```

This creates a `.vscode/settings.json` file with optimal LaTeX configuration.

#### Manual Configuration

If you prefer manual configuration, add the following to VSCode's settings.json:

```json
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
"latex-workshop.message.error.show": true,
"latex-workshop.message.warning.show": true
```

## Building the CV

Once set up, building is straightforward:

1. Open `main.tex` in VSCode/Cursor
2. Either:
   - Click the "Build LaTeX" button (green play button in the LaTeX Workshop tab)
   - Save the file (auto-build is enabled in our configuration)
   - Use the keyboard shortcut (Ctrl+Alt+B or Cmd+Alt+B)
3. View the PDF directly in VSCode (split screen view is recommended)

## Troubleshooting

### Common Issues

- **Missing Packages**: If you see errors about missing packages, install them using your LaTeX distribution's package manager:
  ```bash
  # For TeX Live
  tlmgr install [package-name]
  ```

- **Build Errors**: Check the LaTeX Workshop output panel for specific error messages

- **PDF Viewer Issues**: If the PDF doesn't display, try changing the viewer setting:
  ```json
  "latex-workshop.view.pdf.viewer": "browser"
  ```
