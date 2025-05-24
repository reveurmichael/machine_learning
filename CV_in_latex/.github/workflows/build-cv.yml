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