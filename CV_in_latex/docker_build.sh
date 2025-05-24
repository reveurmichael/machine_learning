#!/bin/bash

# Build the Docker image
docker compose build

# Run the container to generate the PDF
docker compose run --rm cv-builder

echo "CV built successfully. PDF file is available as main.pdf" 