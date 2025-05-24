# Docker-Based LaTeX CV Builder

This guide explains how to build your LaTeX CV using Docker, eliminating the need to install LaTeX locally.

## Advantages of Docker-Based Builds

- **Consistency**: Same build environment regardless of host OS
- **Zero LaTeX Installation**: No need to install and maintain LaTeX locally
- **Cross-Platform**: Works identically on Windows, macOS, and Linux
- **Dependency Management**: All required packages are pre-installed in the container
- **Isolation**: Build process doesn't affect your local system

## Prerequisites

- Docker and Docker Desktop installed on your system
- For Windows, WSL 2 is required

## Building Methods

### Option 1: Using the Python Script (Cross-Platform)

The Python script works on all operating systems and handles error checking:

```bash
python docker_build.py
```

### Option 2: Using the Shell Script (Unix Systems)

For Unix-based systems (Linux, macOS), the shell script provides a native alternative:

```bash
./docker_build.sh
```

### Option 3: Using Docker Compose Directly

For manual control over the build process:

```bash
# Build the Docker image
docker compose build

# Run the container to generate the PDF
docker compose run --rm cv-builder
```

### Option 4: Using Docker CLI Directly

If Docker Compose is unavailable:

```bash
# Build the Docker image
docker build -t latex-cv-builder .

# Run the container to generate the PDF
docker run --rm -v $(pwd):/latex latex-cv-builder
```

All methods will output the PDF to the current directory as `main.pdf`.

## VSCode Integration with DevContainer

For the best development experience, you can use VSCode's DevContainer feature to edit and build your CV in a containerized environment with full LaTeX support.

### DevContainer Setup

1. Install the required VSCode extensions:
   - [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
   - [LaTeX Workshop](https://marketplace.visualstudio.com/items?itemName=James-Yu.latex-workshop)

2. Run our configuration script to set up the DevContainer:
   ```bash
   python config_vscode_devcontainer.py
   ```

3. Open the repository in VSCode

4. Click the green button in the bottom-left corner (or press F1 and select "Dev Containers: Reopen in Container")

5. VSCode will:
   - Build the Docker container
   - Connect to it
   - Configure LaTeX Workshop inside the container

6. Now you can:
   - Edit LaTeX files with full syntax highlighting and autocomplete
   - Save to automatically trigger builds
   - View the PDF preview directly in VSCode
   - Get real-time error feedback

## Customization

### Dockerfile Customization

To add additional LaTeX packages:

1. Edit the Dockerfile:
   ```dockerfile
   # Add packages to this line
   RUN tlmgr install \
       moderncv \
       ulem \
       xcolor \
       # Add your packages here
   ```

2. Rebuild the Docker image:
   ```bash
   docker compose build
   ```


## Troubleshooting

### Docker Issues

- **Docker Not Running**: Make sure Docker Desktop is running
  ```bash
  # Check if Docker is running
  docker info
  ```

- **Permission Issues**: On Linux, you may need to add your user to the docker group
  ```bash
  sudo usermod -aG docker $USER
  # Log out and back in for changes to take effect
  ```

- **Windows-Specific**: Ensure WSL 2 is properly configured for Docker Desktop
