import os
from typing import Optional, Dict, Any
from dotenv import dotenv_values, load_dotenv
from .decorator import *

def get_env_file_path() -> str:
    return ".env"


def load_environment_secrets() -> Dict[str, Any]:
    return dotenv_values(get_env_file_path())


def initialize_environment() -> None:
    load_dotenv(get_env_file_path())


def get_environment_variable(key: str) -> Optional[str]:
    return load_environment_secrets().get(key)


def is_env_file_present() -> bool:
    return os.path.isfile(get_env_file_path())


def is_2captcha_api_key_present() -> bool:
    return get_environment_variable("TWOCAPTCHA_API_KEY") is not None


def display_all_environment_variables() -> None:
    for key, value in load_environment_secrets().items():
        print(f"{key}: {value}")
