## Demo Repo 
https://github.com/reveurmichael/cv_latex

## Template

You can also explore other templates:
- https://www.overleaf.com/gallery/tagged/cv

and then donwload the source code and open it in VSCode/Cursor.

## LaTeX Workshop extension for VSCode/Cursor

Install the **[LaTeX Workshop](https://marketplace.visualstudio.com/items?itemName=James-Yu.latex-workshop)** extension for VSCode/Cursor.

Then, install TexLive:

- https://github.com/James-Yu/LaTeX-Workshop/wiki/Install


### On MacOS:

```bash
brew install texlive
```

### On Windows:
- Install [TexLive](https://www.tug.org/texlive/) 
- Install [Perl](https://strawberryperl.com/) 
- Hit `Ctrl + Shift + P` and type `settings`, choose `Preferences: Open User Settings (JSON)` to open the `settings.json` file.
- Add the following to the `settings.json` file:

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

## Compiling Your CV

1. Open your `.tex` file in VS Code/Cursor
2. Either:
   - Click the "Build LaTeX" button (green play button), or
   - Save the file to trigger automatic PDF generation

## Continuous Integration

### GitHub Actions Setup

Create a file at `.github/workflows/build-cv.yml` with the following content:

```yml
name: Build LaTeX CV

on:
  push:
    branches:
      - main

jobs:
  build-and-release:
    runs-on: ubuntu-latest
    permissions:
      contents: write
    steps:
      - name: Checkout repository
        uses: actions/checkout@v3
        with:
          fetch-depth: 0

      - name: Set up LaTeX
        uses: xu-cheng/latex-action@v2
        with:
          root_file: main.tex

      - name: Prepare PDF for release branch
        run: |
          mkdir -p /tmp/cv_release
          cp main.pdf /tmp/cv_release/main.pdf

      - name: Create release branch with only PDF
        run: |
          git config --local user.email "action@github.com"
          git config --local user.name "GitHub Action"
          git checkout --orphan release
          # Remove all files and folders except .git
          find . -mindepth 1 -maxdepth 1 ! -name '.git' ! -name '.' -exec rm -rf {} +
          cp /tmp/cv_release/main.pdf .
          git add main.pdf
          git commit -m "Update CV PDF"
          git push -f origin release

      - name: Create GitHub Release
        id: create_release
        uses: actions/create-release@v1
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        with:
          tag_name: v${{ github.run_number }}
          release_name: Release v${{ github.run_number }}
          body: "Automated CV PDF build."
          draft: false
          prerelease: false

      - name: Upload Release Asset
        uses: actions/upload-release-asset@v1
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        with:
          upload_url: ${{ steps.create_release.outputs.upload_url }}
          asset_path: ./main.pdf
          asset_name: main.pdf
          asset_content_type: application/pdf 
```

**Note:** You may need to modify this workflow based on your specific template structure and file names.

## Recommended .gitignore

```
# LaTeX temporary files
*.aux
*.lof
*.log
*.lot
*.fls
*.out
*.toc
*.fmt
*.fot
*.cb
*.cb2
*.ptc
.*.lb
*.bbl
*.bcf
*.blg
*-blx.aux
*-blx.bib
*.run.xml
*.fdb_latexmk
*.synctex
*.synctex(busy)
*.synctex.gz
*.synctex.gz(busy)
*.pdfsync
latex.out/

# Algorithm files
*.alg
*.loa

# Generated PDFs
*.pdf

# Other common ignore patterns
~*
~*.pptx
~*.docx
~*.pdf
*/_build/**
__pycache__
**/__pycache__/**
.pytest_cache
.vscode
.idea
/cmake-build-debug/
/out/
/experiments/
~$*
*~
*.swp
*.swo
.ipynb_checkpoints
*.pyc
/venv/
*.py[cod]
*.DS_Store
*/tmp/*
/tmp/
*/tmp/**
*/**/node_modules/*
*/node_modules/*
*/.python-version
.python-version
```