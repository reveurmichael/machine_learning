## LaTeX 

### Template

The template is based on the IEEE conference template, hosted also on Overleaf:

- https://www.overleaf.com/latex/templates/ieee-conference-template/grfzhhncsfqn


You can also explore other templates:
- https://www.overleaf.com/latex/templates/ieee-for-journals-template-with-bibtex-example-files-included/hjbyjvncdmpx


### To learn more about LaTex

Example papers with LaTeX source:

- https://www.overleaf.com/read/xyrcxdngnrsg#b69338
- https://www.overleaf.com/read/kkfgcrzkcfjq#4c3583


### LaTeX Workshop extension for VSCode/Cursor

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


### Project Structure

The main entry point for this LaTeX project, located in the `Latex` folder, is:

- `conference_101719.tex`

You can rename it to `my_paper.tex`, `zhuxinning.tex` or anything you want.


### Compilation of Tex file

1. Open your `.tex` file in VSCode/Cursor
1. Click the "Build LaTeX" button (green play button)
1. PDF output will be generated automatically

Or, even better, on saving the file, the PDF will be generated automatically.


## Git 

Git commit and push often.

Always git pull before starting new work.

