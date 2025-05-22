import os
import tempfile
import logging
from typing import Optional

from shell import ShellExecutor

class CodeUtils:
    """
    Streamlined code utilities for essential operations when interacting with LLMs.
    Focused on executing code and managing files for the AgenticVM.
    """
    
    def __init__(self, shell_executor=None, config_path=None):
        """
        Initialize code utilities with shell executor
        
        Args:
            shell_executor: Shell executor instance for running commands
            config_path: Path to configuration file (not used directly)
        """
        try:
            # Setup logging
            self.logger = logging.getLogger("CodeUtils")
            self.logger.setLevel(logging.INFO)
            
            # Initialize shell executor
            self.shell = shell_executor or ShellExecutor()
            
            # Log initialization
            self.logger.info("CodeUtils initialized successfully")
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"Error initializing CodeUtils: {str(e)}")
            print(f"Error initializing CodeUtils: {str(e)}")
    
    def create_file(self, file_path: str, content: str) -> str:
        """
        Create a new file with the specified content
        
        Args:
            file_path: Path where the file should be created
            content: Content to write to the file
            
        Returns:
            Result message from the operation
        """
        try:
            self.logger.info(f"Creating file at {file_path}")
            return self.shell.write_file(file_path, content)
        except Exception as e:
            error_msg = f"Error creating file: {str(e)}"
            self.logger.error(error_msg)
            return error_msg
    
    def read_file(self, file_path: str) -> str:
        """
        Read content from a file
        
        Args:
            file_path: Path to the file to read
            
        Returns:
            File content or error message
        """
        try:
            self.logger.info(f"Reading file from {file_path}")
            return self.shell.read_file(file_path)
        except Exception as e:
            error_msg = f"Error reading file: {str(e)}"
            self.logger.error(error_msg)
            return error_msg
    
    def run_code(self, code: str, timeout: int = 30) -> str:
        """
        Run Python code and capture the output
        
        Args:
            code: The code to run
            timeout: Maximum execution time in seconds
            
        Returns:
            Output from running the code
        """
        try:
            # Create temporary file with the code
            with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode='w') as f:
                f.write(code)
                temp_path = f.name
            
            self.logger.info(f"Running code from temporary file: {temp_path}")
            
            # Run the code using shell executor
            result = self.shell.execute(f"python {temp_path}", timeout=timeout)
            return result
        except Exception as e:
            error_msg = f"Error running code: {str(e)}"
            self.logger.error(error_msg)
            return error_msg
        finally:
            try:
                if 'temp_path' in locals():
                    os.unlink(temp_path)
            except Exception as e:
                self.logger.warning(f"Could not delete temporary file: {str(e)}") 