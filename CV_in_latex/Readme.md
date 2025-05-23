## Demo Repo 
https://github.com/reveurmichael/cv_latex

## Template

You can also explore other templates:
- https://www.overleaf.com/gallery/tagged/cv

## LaTeX Workshop extension for VSCode/Cursor

Install the **LaTeX Workshop** extension for VSCode/Cursor.

Then, install TexLive:

- https://github.com/James-Yu/LaTeX-Workshop/wiki/Install


On MacOS:

```bash
brew install texlive
```

On Windows:
- Install TexLive 
- Install Perl 
- Add the following to VS Code settings.json:

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

## Compilation of Tex file

1. Open your `.tex` file in VSCode/Cursor
1. Click the "Build LaTeX" button (green play button)
1. PDF output will be generated automatically

Or, even better, on saving the file, the PDF will be generated automatically.

## `.gitignore`

```
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
# algorithms
*.alg
*.loa
*.pdf
```

## GitHub Actions 

Create a file, in `.github/workflows/build-cv.yml`, with the content

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

If you are using the template from the demo repo, you can just copy the file and rename it to `build-cv.yml`. For other templates, you might need to twist a little bit the yml file.