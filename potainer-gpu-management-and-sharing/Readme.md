### 🎓 Goal of the Session

Train students to:

* Understand GPU containerization & access control
* Set up secure, SSH-based user access
* Use Portainer to manage GPU containers (start/stop/monitor)
* Troubleshoot container-level GPU usage


### 🧠 Teaching Plan (4-Hour Session Outline)

#### 🕐 **Hour 1: Introduction + Setup**

* **Theory**: What is Portainer? Why use it for GPU cluster management?
* **Hands-on**:

  * Provision a VM/server with an NVIDIA GPU and Docker installed
  * Pull and run the Portainer CE container with Docker
  * Mount `nvidia-docker2` and verify GPU passthrough with:

    ```bash
    docker run --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
    ```
  * Open Portainer in the browser

#### 🕑 **Hour 2: SSH Access & Docker Contexts**

* **Each student gets their own Linux user** with:

  * Passwordless sudo for Docker (or access to `docker.sock`)
  * Their own `.ssh/authorized_keys` entry
* Show them how to:

  * SSH into their user
  * Use `docker context` to switch between local and remote hosts
* Optional: Introduce shared user groups (`docker`, `gpuusers`)

> 🎯 *Goal: Each student can SSH into a Linux server and control containers via CLI or Portainer.*

#### 🕒 **Hour 3: GPU Containers via Portainer**

* Students log in to Portainer (admin or user-permission view)
* Launch GPU-enabled containers using:

  * The `nvidia/cuda` base images
  * Or Jupyter containers with GPU support
* Explore:

  * Resource monitoring (GPU usage per container)
  * Volumes and bind mounts for persistent notebooks
  * Environment variables (e.g., `CUDA_VISIBLE_DEVICES`)

### 🧰 Tools & Stack

| Tool                     | Purpose                       |
| ------------------------ | ----------------------------- |
| Docker + NVIDIA Docker   | Containerization + GPU access |
| Portainer CE             | UI-based Docker orchestration |
| SSH + user management    | Secure student logins         |
| NVIDIA Container Toolkit | GPU pass-through              |
| `nvidia-smi`, `htop`     | Monitoring usage              |


### ✅ Learning Outcomes

By the end of the session, students will:

* Know how to manage GPU containers securely
* Use both CLI and Portainer for orchestration
* Handle basic GPU debugging and job management
* Be prepared to work as cluster admins or DevOps assistants in ML teams
