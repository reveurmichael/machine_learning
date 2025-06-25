import os
import subprocess
from typing import Optional
from shell import ShellExecutor


class PipManager:
    """
    A streamlined Python package manager with essential functionality
    for managing Python packages.
    """
    
    def __init__(self, shell_executor: Optional[ShellExecutor] = None):
        """Initialize the Pip manager"""
        self.shell = shell_executor or ShellExecutor()
        
        # Detect pip command (pip or pip3)
        self.pip_cmd = self._detect_pip_command()
    
    def _detect_pip_command(self) -> str:
        """Detect which pip command is available (pip or pip3)"""
        try:
            # Try pip3 first
            result = subprocess.run(["pip3", "--version"], 
                                  stdout=subprocess.PIPE, 
                                  stderr=subprocess.PIPE)
            if result.returncode == 0:
                return "pip3"
        except FileNotFoundError:
            pass
            
        try:
            # Try pip next
            result = subprocess.run(["pip", "--version"], 
                                  stdout=subprocess.PIPE, 
                                  stderr=subprocess.PIPE)
            if result.returncode == 0:
                return "pip"
        except FileNotFoundError:
            pass
            
        # Default to pip if we couldn't determine
        return "pip"
    
    def install(self, package: str, version: Optional[str] = None) -> str:
        """
        Install a Python package
        
        Args:
            package (str): Package name to install
            version (str, optional): Specific version to install
            
        Returns:
            str: Installation result
        """
        # Build the command
        cmd = [self.pip_cmd, "install"]
        
        # Add the package with optional version
        if version:
            cmd.append(f"{package}=={version}")
        else:
            cmd.append(package)
        
        # Execute the command
        return self.shell.execute(" ".join(cmd))
    
    def uninstall(self, package: str, yes: bool = True) -> str:
        """
        Uninstall a Python package
        
        Args:
            package (str): Package name to uninstall
            yes (bool): Whether to automatically confirm uninstallation
            
        Returns:
            str: Uninstallation result
        """
        # Build the command
        cmd = [self.pip_cmd, "uninstall"]
        
        if yes:
            cmd.append("-y")
        
        cmd.append(package)
        
        # Execute the command
        return self.shell.execute(" ".join(cmd))
    
    def list_packages(self) -> str:
        """
        List installed packages
        
        Returns:
            str: List of installed packages
        """
        # Build the command
        cmd = f"{self.pip_cmd} list"
        
        # Execute the command
        return self.shell.execute(cmd)
    
    def install_requirements(self, requirements_file: str) -> str:
        """
        Install packages from a requirements file
        
        Args:
            requirements_file (str): Path to requirements file
            
        Returns:
            str: Installation result
        """
        # Build the command
        cmd = [self.pip_cmd, "install", "-r", requirements_file]
        
        # Execute the command
        return self.shell.execute(" ".join(cmd))
