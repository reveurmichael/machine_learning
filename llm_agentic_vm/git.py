import os
import subprocess
import logging
import yaml
from typing import Dict, Optional
from shell import ShellExecutor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class GitManager:
    """
    A streamlined Git manager that provides essential functionality for
    working with Git repositories in the context of LLM interactions.
    """
    
    def __init__(self, shell_executor=None, config_path=None):
        """
        Initialize the Git manager
        
        Args:
            shell_executor: Optional ShellExecutor instance
            config_path: Path to config.yml file
        """
        self.shell = shell_executor or ShellExecutor()
        self.logger = logging.getLogger("GitManager")
        self.current_repo = None
        
        # Default git configuration with SSH
        self.git_config = {
            "use_ssh": True,
            "ssh_key_path": "~/.ssh/id_rsa",
            "default_user_name": "AgenticVM User",
            "default_user_email": "user@example.com"
        }
        
        # Load configuration if provided
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    full_config = yaml.safe_load(f) or {}
                git_config = full_config.get('git', {})
                if git_config:
                    self.git_config.update(git_config)
            except Exception as e:
                self.logger.error(f"Error loading git configuration: {e}")
        
        # Apply git configuration
        if self.git_config.get("use_ssh", True):
            # Configure SSH for git
            ssh_key_path = os.path.expanduser(self.git_config.get("ssh_key_path", "~/.ssh/id_rsa"))
            if os.path.exists(ssh_key_path):
                self.logger.info(f"Using SSH key at {ssh_key_path}")
                # Set GIT_SSH_COMMAND environment variable
                os.environ["GIT_SSH_COMMAND"] = f"ssh -i {ssh_key_path} -o IdentitiesOnly=yes"
            else:
                self.logger.warning(f"SSH key not found at {ssh_key_path}, falling back to default authentication")
        
    def clone_repository(self, url: str, directory: Optional[str] = None) -> str:
        """
        Clone a git repository
        
        Args:
            url (str): URL of the repository to clone
            directory (str, optional): Directory to clone into
            
        Returns:
            str: Command output
        """
        self.logger.info(f"Cloning repository: {url}")
        
        # Convert HTTPS URL to SSH if needed
        if self.git_config.get("use_ssh", True) and url.startswith("https://github.com/"):
            url = url.replace("https://github.com/", "git@github.com:")
            if not url.endswith(".git"):
                url += ".git"
        
        cmd = ["git", "clone"]
        cmd.append(url)
        
        if directory:
            cmd.append(directory)
            self.current_repo = directory
        else:
            # Extract repo name from URL to set current_repo
            repo_name = url.split('/')[-1]
            if repo_name.endswith('.git'):
                repo_name = repo_name[:-4]
            self.current_repo = repo_name
        
        return self.shell.execute(" ".join(cmd))
    
    def git_init(self, directory: str = ".") -> str:
        """
        Initialize a git repository
        
        Args:
            directory (str): Directory to initialize
            
        Returns:
            str: Command output
        """
        self.logger.info(f"Initializing git repository in {directory}")
        
        if directory != ".":
            # Check if directory exists, create if not
            if not os.path.exists(directory):
                os.makedirs(directory)
            
            cmd = f"cd {directory} && git init"
            self.current_repo = directory
        else:
            cmd = "git init"
            self.current_repo = os.getcwd()
        
        return self.shell.execute(cmd)
    
    def git_add(self, path: str = ".") -> str:
        """
        Add files to git staging area
        
        Args:
            path (str): Path to add
            
        Returns:
            str: Command output
        """
        self.logger.info(f"Adding files to git staging area: {path}")
        return self.shell.execute(f"git add {path}")
    
    def git_commit(self, message: str) -> str:
        """
        Commit changes to git
        
        Args:
            message (str): Commit message
            
        Returns:
            str: Command output
        """
        self.logger.info(f"Committing changes: {message}")
        return self.shell.execute(f"git commit -m \"{message}\"")
    
    def git_push(self, remote: str = "origin", branch: str = "main") -> str:
        """
        Push changes to remote git repository
        
        Args:
            remote (str): Remote name
            branch (str): Branch name
            
        Returns:
            str: Command output
        """
        self.logger.info(f"Pushing to {remote}/{branch}")
        return self.shell.execute(f"git push {remote} {branch}")
    
    def git_pull(self, remote: str = "origin", branch: str = "main") -> str:
        """
        Pull changes from remote git repository
        
        Args:
            remote (str): Remote name
            branch (str): Branch name
            
        Returns:
            str: Command output
        """
        self.logger.info(f"Pulling from {remote}/{branch}")
        return self.shell.execute(f"git pull {remote} {branch}")
    
    def git_status(self) -> str:
        """
        Check git repository status
            
        Returns:
            str: Status output
        """
        self.logger.info("Checking git status")
        return self.shell.execute("git status") 