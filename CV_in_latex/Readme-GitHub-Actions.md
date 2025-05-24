# Automated CV Building with GitHub Actions

This guide explains how to leverage GitHub Actions for continuous integration and automated PDF generation for your LaTeX CV.

## Advantages of GitHub Actions

- **Full Automation**: Your CV is built automatically on every push
- **Version Control**: Each change is tracked and archived
- **PDF Releases**: Automatically generate PDF releases with version numbers
- **No Local Setup**: No need for LaTeX or Docker on your local machine
- **CI/CD Integration**: Part of a professional development workflow

## How It Works

When you push changes to the `main` or `master` branch, GitHub Actions:

1. Sets up a LaTeX environment in the cloud
2. Compiles your CV into a PDF
3. Creates a release branch containing the PDF
4. Generates a GitHub Release with the PDF attached

## Setup Process

The repository already includes the necessary workflow configuration at `.github/workflows/build-cv.yml`.

### Understanding the Workflow File

The workflow file contains the following key components:

```yml
name: Build LaTeX CV

on:
  push:
    branches:
      - main
      - master

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

## Troubleshooting
1. Go to the "Actions" tab in your repository
2. Look for the latest workflow run
3. Click on it to see detailed logs
4. If there are errors, expand the failing step for details
