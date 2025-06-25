import os
import sys
import subprocess
import tempfile
import logging
from typing import Dict, List, Optional
from pathlib import Path

from shell import ShellExecutor


class PythonManager:
    """
    Streamlined Python manager for AgenticVM.
    Focused on executing Python code and managing Python environments.
    """
    
    def __init__(self, shell_executor: Optional[ShellExecutor] = None):
        """
        Initialize the Python manager
        
        Args:
            shell_executor: Optional shell executor instance
        """
        try:
            # Setup logging
            self.logger = logging.getLogger("PythonManager")
            self.logger.setLevel(logging.INFO)
            
            # Get or create shell executor
            self.shell = shell_executor or ShellExecutor()
            
            # Detect python command (python or python3)
            self.python_cmd = self._detect_python_command()
            
            # Detect pip command (pip or pip3)
            self.pip_cmd = self._detect_pip_command()
            
            self.logger.info(f"Initialized with python_cmd={self.python_cmd}, pip_cmd={self.pip_cmd}")
        except Exception as e:
            self.logger.error(f"Error initializing PythonManager: {str(e)}")
            # Default values if detection fails
            self.python_cmd = "python"
            self.pip_cmd = "pip"
    
    def _detect_python_command(self) -> str:
        """
        Detect which Python command is available (python or python3)
        
        Returns:
            The available Python command
        """
        try:
            # Try python3 first
            result = subprocess.run(
                ["python3", "--version"], 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return "python3"
        except (FileNotFoundError, subprocess.SubprocessError, Exception) as e:
            self.logger.debug(f"python3 command check failed: {str(e)}")
            
        try:
            # Try python next
            result = subprocess.run(
                ["python", "--version"], 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return "python"
        except (FileNotFoundError, subprocess.SubprocessError, Exception) as e:
            self.logger.debug(f"python command check failed: {str(e)}")
            
        # Default to python if we couldn't determine
        return "python"
    
    def _detect_pip_command(self) -> str:
        """
        Detect which pip command is available (pip or pip3)
            
        Returns:
            The available pip command
        """
        try:
            # Try pip3 first
            result = subprocess.run(
                ["pip3", "--version"], 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return "pip3"
        except (FileNotFoundError, subprocess.SubprocessError, Exception) as e:
            self.logger.debug(f"pip3 command check failed: {str(e)}")
            
        try:
            # Try pip next
            result = subprocess.run(
                ["pip", "--version"], 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return "pip"
        except (FileNotFoundError, subprocess.SubprocessError, Exception) as e:
            self.logger.debug(f"pip command check failed: {str(e)}")
            
        # Default to pip if we couldn't determine
        return "pip"
    
    def execute_code(self, code: str, timeout: int = 30) -> str:
        """
        Execute Python code and return the output
        
        Args:
            code: Python code to execute
            timeout: Timeout in seconds
            
        Returns:
            Output of the code execution
        """
        try:
            # Create a temporary file with the code
            with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as f:
                f.write(code)
                temp_file = f.name
            
            # Execute the file
            self.logger.info(f"Executing Python code from temporary file: {temp_file}")
            result = self.shell.execute(f"{self.python_cmd} {temp_file}", timeout=timeout)
            return result
        except Exception as e:
            error_msg = f"Error executing Python code: {str(e)}"
            self.logger.error(error_msg)
            return error_msg
        finally:
            # Clean up the temporary file
            try:
                if 'temp_file' in locals():
                    os.unlink(temp_file)
            except Exception as e:
                self.logger.warning(f"Could not delete temporary file: {str(e)}")
    
    def execute_script(self, script_path: str, args: Optional[List[str]] = None) -> str:
        """
        Execute a Python script
        
        Args:
            script_path: Path to the script
            args: Arguments to pass to the script
            
        Returns:
            Script output
        """
        try:
            # Validate script exists
            if not os.path.isfile(script_path):
                return f"Error: Script not found: {script_path}"
            
            args = args or []
            cmd_parts = [self.python_cmd, script_path] + args
            cmd = " ".join(cmd_parts)
            
            self.logger.info(f"Executing script: {cmd}")
            return self.shell.execute(cmd)
        except Exception as e:
            error_msg = f"Error executing script: {str(e)}"
            self.logger.error(error_msg)
            return error_msg
    
    def get_python_version(self) -> str:
        """
        Get the Python version
            
        Returns:
            Python version string
        """
        try:
            return self.shell.execute(f"{self.python_cmd} --version")
        except Exception as e:
            error_msg = f"Error getting Python version: {str(e)}"
            self.logger.error(error_msg)
            return error_msg
    
    def install_package(self, package: str, version: Optional[str] = None) -> str:
        """
        Install a Python package
        
        Args:
            package: Package name
            version: Optional package version
            
        Returns:
            Installation result
        """
        try:
            # Sanitize package name
            package = package.strip()
            if not package:
                return "Error: Empty package name"
            
            # Construct the command
            cmd_parts = [self.pip_cmd, "install", "--upgrade"]
            
            if version:
                cmd_parts.append(f"{package}=={version}")
            else:
                cmd_parts.append(package)
            
            cmd = " ".join(cmd_parts)
            
            # Execute the command
            self.logger.info(f"Installing package: {cmd}")
            return self.shell.execute(cmd)
        except Exception as e:
            error_msg = f"Error installing package: {str(e)}"
            self.logger.error(error_msg)
            return error_msg
