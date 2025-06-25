import subprocess
import os
import tempfile
import logging
import shlex
from typing import Dict, Optional, Tuple, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class ShellExecutor:
    """
    Versatile shell command executor for AgenticVM.
    Provides secure command execution with proper error handling.
    """
    def __init__(self):
        self.working_directory = os.getcwd()
        self.env_vars = os.environ.copy()
        self.logger = logging.getLogger("ShellExecutor")
        self.secure_mode = False
        
        # Commands that are potentially dangerous and should be blocked in secure mode
        self.blocked_commands = [
            "rm -rf /", "rm -rf /*", "mkfs", "dd if=/dev/zero",
            "> /dev/sda", "chmod -R 777 /", ":(){:|:&};:", "sudo rm", 
            "> /dev/null", "mv /* ", "find / -delete"
        ]
        
        # Initialize debug log
        try:
            log_dir = os.path.join(self.working_directory, "logs")
            os.makedirs(log_dir, exist_ok=True)
            self.debug_log_path = os.path.join(log_dir, "shell_debug.log")
        except Exception as e:
            self.logger.warning(f"Could not create logs directory: {str(e)}")
            self.debug_log_path = "/tmp/shell_debug.log"
    
    def resolve_path(self, path: str) -> str:
        """
        Resolve special path elements like ~ (home directory)
        
        Args:
            path: Path string that may contain ~ or other special elements
            
        Returns:
            Resolved absolute path
        """
        try:
            # Expand user directory (~/something becomes /home/user/something)
            expanded_path = os.path.expanduser(path)
            
            # Make path absolute if it's not already
            if not os.path.isabs(expanded_path):
                expanded_path = os.path.abspath(expanded_path)
                
            return expanded_path
        except Exception as e:
            self.logger.error(f"Path resolution error: {str(e)}")
            return path  # Return original path if resolution fails
    
    def execute(self, command: str, timeout: int = 60, env: Optional[Dict[str, str]] = None) -> str:
        """
        Execute a shell command and return the output
        
        Args:
            command (str): The shell command to execute
            timeout (int): Maximum execution time in seconds
            env (dict): Environment variables for the command
            
        Returns:
            str: Command output including stdout and stderr
        """
        try:
            self.logger.info(f"Executing command: {command}")
            
            # Validate and clean command
            command = self._clean_command(command)
            if not command:
                return "Error: Empty or invalid command"
            
            # Log command details
            self._log_debug(f"Command: {command}")
            self._log_debug(f"Working directory: {self.working_directory}")
            
            # Security checks
            if self._is_potentially_dangerous(command):
                error_msg = f"Command blocked for security reasons: {command}"
                self.logger.warning(error_msg)
                return error_msg
            
            # Prepare environment variables
            cmd_env = self.env_vars.copy()
            if env:
                cmd_env.update(env)
            
            # Log current working directory
            self.logger.info(f"Working directory: {self.working_directory}")
            
            # Execute the command
            self.logger.info(f"Starting subprocess for command: {command}")
            self._log_debug("Starting subprocess")
            
            # Using subprocess.run with capture output for better control
            try:
                # Create a temp directory for subprocess if needed
                if not os.path.exists(self.working_directory):
                    temp_dir = tempfile.mkdtemp()
                    self.logger.warning(f"Working directory not found, using temp dir: {temp_dir}")
                    working_dir = temp_dir
                else:
                    working_dir = self.working_directory
                
                # Run the command with output capture
                process = subprocess.run(
                    command,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=timeout,
                    cwd=working_dir,
                    env=cmd_env
                )
                
                # Combine stdout and stderr
                output = ""
                if process.stdout:
                    output += process.stdout
                if process.stderr:
                    if output:
                        output += "\n"
                    output += process.stderr
                
                # Log results
                self.logger.info(f"Command completed with exit code: {process.returncode}")
                self._log_debug(f"Exit code: {process.returncode}")
                if process.returncode != 0:
                    self._log_debug(f"Error output: {process.stderr}")
                
                return output if output else f"Command executed with exit code: {process.returncode}"
            except subprocess.TimeoutExpired:
                error_msg = f"Command timed out after {timeout} seconds"
                self.logger.error(f"{error_msg}: {command}")
                return error_msg
            except Exception as sub_e:
                error_msg = f"Error in subprocess: {str(sub_e)}"
                self.logger.error(error_msg)
                self._log_debug(f"Subprocess error: {str(sub_e)}")
                return error_msg
        except Exception as e:
            error_msg = f"Error executing command: {str(e)}"
            self.logger.error(error_msg)
            self._log_debug(error_msg)
            return error_msg

    def _clean_command(self, command: str) -> str:
        """
        Clean and validate a command string
        
        Args:
            command: Raw command string
            
        Returns:
            Cleaned command string or empty string if invalid
        """
        if not command or not isinstance(command, str):
            return ""
        
        # Handle multi-line commands (take only first line for safety)
        if '\n' in command:
            command = command.strip().split('\n')[0]
            self.logger.warning(f"Multi-line command detected. Using only first line: {command}")
        
        # Trim whitespace
        command = command.strip()
        
        # Check if command starts with a number followed by a period (likely a numbered list item)
        if command and command[0].isdigit() and len(command) > 1 and command[1] == '.':
            self.logger.warning(f"Invalid command format. Command appears to be a numbered list item: {command}")
            return ""
        
        return command
    
    def _is_potentially_dangerous(self, command: str) -> bool:
        """
        Check if a command is potentially dangerous
        
        Args:
            command: Command to check
            
        Returns:
            True if command appears dangerous, False otherwise
        """
        if not self.secure_mode:
            return False
            
        command_lower = command.lower().strip()
        
        # Check against blocklist
        for blocked in self.blocked_commands:
            if blocked in command_lower:
                return True
        
        # Additional security checks
        dangerous_patterns = [
            "rm -rf", "format", "mkfs", "dd", "chmod -R", "chown -R",
            "wget", "curl", "> /dev/", "|rm", ";rm", "&&rm", "|| rm"
        ]
        
        if any(pattern in command_lower for pattern in dangerous_patterns):
            # If potentially dangerous, require explicit confirmation in secure mode
            return True
            
        return False
    
    def _log_debug(self, message: str) -> None:
        """Log debug information to file"""
        try:
            with open(self.debug_log_path, 'a') as f:
                f.write(f"{message}\n")
        except Exception:
            pass  # Fail silently
    
    def change_directory(self, path: str) -> str:
        """
        Change the working directory
        
        Args:
            path: New working directory
            
        Returns:
            Result message
        """
        try:
            # Resolve the path
            resolved_path = self.resolve_path(path)
            
            # Check if the directory exists
            if not os.path.isdir(resolved_path):
                return f"Directory does not exist: {path}"
            
            # Change the working directory
            os.chdir(resolved_path)
            self.working_directory = resolved_path
            
            return f"Changed directory to {resolved_path}"
        except Exception as e:
            error_msg = f"Error changing directory: {str(e)}"
            self.logger.error(error_msg)
            return error_msg
    
    def write_file(self, file_path: str, content: str) -> str:
        """
        Write content to a file
        
        Args:
            file_path: Path to the file
            content: Content to write
            
        Returns:
            Success message or error
        """
        try:
            # Resolve the file path
            resolved_path = self.resolve_path(file_path)
            
            # Create parent directories if they don't exist
            os.makedirs(os.path.dirname(resolved_path), exist_ok=True)
            
            # Write content to file
            with open(resolved_path, 'w') as f:
                f.write(content)
            
            return f"File written to {file_path}"
        except Exception as e:
            error_msg = f"Error writing file: {str(e)}"
            self.logger.error(error_msg)
            return error_msg
    
    def read_file(self, file_path: str) -> str:
        """
        Read content from a file
        
        Args:
            file_path: Path to the file
            
        Returns:
            File content or error message
        """
        try:
            # Resolve the file path
            resolved_path = self.resolve_path(file_path)
            
            # Read file content
            with open(resolved_path, 'r') as f:
                content = f.read()
            
            return content
        except Exception as e:
            error_msg = f"Error reading file: {str(e)}"
            self.logger.error(error_msg)
            return error_msg