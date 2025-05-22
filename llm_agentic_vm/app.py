import streamlit as st
import os
import json
import yaml
import time
import logging
from dotenv import load_dotenv
from shell import ShellExecutor
from code_utils import CodeUtils
from llm_client import LLMClientManager
from json_utils import parse_command_response
from config import SYSTEM_PROMPT, get_workspace_path, LOGGING_CONFIG

# Configure logging
logger = logging.getLogger("app")
logger.setLevel(logging.INFO)
formatter = logging.Formatter(LOGGING_CONFIG["log_format"])

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Load environment variables
load_dotenv()

# Initialize authentication states
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
    
# Add logger to session state for other components to use
if "logger" not in st.session_state:
    st.session_state.logger = logger

# Login functionality
if not st.session_state.authenticated:
    st.title("AgenticVM Login")
    
    # Default credentials (should be changed in production)
    default_username = "admin"
    default_password = "password"
    
    # Login form
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")
        
        if submit:
            if username == default_username and password == default_password:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Invalid username or password")
    
    # Stop execution here if not authenticated
    st.stop()

# Only execute below this point if user is authenticated
# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "executor" not in st.session_state:
    st.session_state.executor = ShellExecutor()
if "code_utils" not in st.session_state:
    config_path = os.path.join(os.path.dirname(__file__), "config.yml")
    st.session_state.code_utils = CodeUtils(st.session_state.executor, config_path=config_path)
if "llm_manager" not in st.session_state:
    # Initialize LLM client manager
    config_path = os.path.join(os.path.dirname(__file__), "config.yml")
    st.session_state.llm_manager = LLMClientManager(config_path if os.path.exists(config_path) else None)

# App title and description
st.title("AgenticVM")
st.markdown("""
This application allows you to manage your cloud virtual machine through natural language.
You can ask to install packages, run commands, write code, execute tasks, and more.
""")

# LLM status indicator
if st.session_state.llm_manager.default_client:
    client_name = st.session_state.llm_manager.default_client
    client = st.session_state.llm_manager.clients.get(client_name)
    if client:
        st.success(f"Using LLM: {client.provider} - {client.model}")
    else:
        st.warning("No LLM configured. Please update config.yml with your LLM settings.")

# Chat interface
chat_container = st.container()

with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Input area with access to utilities
prompt = st.chat_input("What would you like to do?")
if prompt:
    # Check if LLM is configured
    llm_clients = st.session_state.llm_manager.list_clients()
    if not llm_clients:
        st.error("Please configure an LLM provider in config.yml")
        st.stop()
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Get AI response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = st.session_state.llm_manager.generate(
                    prompt=prompt,
                    system_message=SYSTEM_PROMPT,
                    history=[{"role": msg["role"], "content": msg["content"]} 
                        for msg in st.session_state.messages[:-1]]
                    )

                
                # Parse the response for command steps using JSON format
                try:
                    success, result = parse_command_response(response)
                    
                    if success and isinstance(result, dict) and "steps" in result:
                        # We have valid JSON steps to execute
                        steps = result["steps"]
                        
                        # Display the response without the JSON
                        full_response = ""
                        
                        # Process each step
                        for i, step in enumerate(steps):
                            try:
                                explanation = step["explanation"]
                                command = step["command"]
                                
                                # Format command with markdown
                                step_text = f"**Step {i+1}**: {explanation}\n\n```bash\n{command}\n```\n\n"
                                st.markdown(step_text)
                                full_response += step_text
                                
                                # Execute the command
                                with st.status(f"Executing step {i+1}..."):
                                    # Check for syntax issues before executing
                                    has_syntax_error = False
                                    error_message = ""
                                    
                                    # Validate quotes
                                    single_quotes = command.count("'")
                                    double_quotes = command.count('"')
                                    backticks = command.count("`")
                                    
                                    if single_quotes % 2 != 0:
                                        has_syntax_error = True
                                        error_message = "Error: Unbalanced single quotes in command"
                                    elif double_quotes % 2 != 0:
                                        has_syntax_error = True
                                        error_message = "Error: Unbalanced double quotes in command"
                                    elif backticks % 2 != 0:
                                        has_syntax_error = True
                                        error_message = "Error: Unbalanced backticks in command"
                                    
                                    if has_syntax_error:
                                        st.error(error_message)
                                        full_response += f"Error: {error_message}\n\n"
                                    else:
                                        # Execute the command
                                        try:
                                            result = st.session_state.executor.execute(command)
                                            st.code(result, language="bash")
                                            full_response += f"Result:\n```\n{result}\n```\n\n"
                                        except Exception as cmd_e:
                                            error_msg = f"Error executing command: {str(cmd_e)}"
                                            st.error(error_msg)
                                            full_response += f"Error: {error_msg}\n\n"
                                            logger.error(f"Command execution failed: {command} - {str(cmd_e)}")
                            except Exception as step_e:
                                error_msg = f"Error processing step {i+1}: {str(step_e)}"
                                st.error(error_msg)
                                full_response += f"Error: {error_msg}\n\n"
                                logger.error(f"Step processing failed: {str(step_e)}")
                        
                        # Store the full response in the chat history
                        st.session_state.messages.append({"role": "assistant", "content": full_response})
                    else:
                        # Regular response with no commands
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        
                        # Log the issue for debugging
                        if not success:
                            logger.warning(f"Could not parse response as JSON: {result}")
                except Exception as parse_e:
                    # Error in JSON parsing
                    logger.error(f"Error parsing LLM response: {str(parse_e)}")
                    st.error(f"Error parsing response: {str(parse_e)}")
                    
                    # Still try to display the response
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                error_message = f"Error generating response: {str(e)}"
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})
