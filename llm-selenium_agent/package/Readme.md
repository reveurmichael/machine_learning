# Selenium Streamlit Setup Helper

A simple Python package that makes setting up Selenium WebDriver with Chrome and Firefox browsers easy. It provides a Streamlit web interface for configuration and testing.

## What this package does

- ✅ Automatically installs Chrome and Firefox WebDrivers
- ✅ Provides a user-friendly Streamlit interface
- ✅ Helps test browser connections
- ✅ Configures headless mode and other settings
- ✅ Makes it easy to start your own Selenium automation projects

## Installation

```bash
cd llm-selenium-agent/package

# Install the package
pip install -e .
```

## Getting Started

### Step 1: Run the first-time setup

This command will set up your environment:

```bash
llm_selenium_agent_first_time_setup
```

It will:
- Create a basic configuration file
- Install Chrome and Firefox WebDrivers
- Create necessary folders and files

### Step 2: Launch the Streamlit app

```bash
llm_selenium_agent_streamlit_app
```

This will open a web interface where you can:
- Verify your WebDriver installations
- Test browser connections to websites
- Configure settings like headless mode

## Example Files

The package includes example files to help you get started:

1. **example_launch_script.py**: A simple script to launch the Streamlit app
   ```bash
   # Copy to your project directory
   python example_launch_script.py
   ```

2. **example_automation.py**: A demonstration of a simple Google search automation
   ```bash
   # Copy to your project directory
   python example_automation.py
   ```

## Creating Your Own Automation Script

```python
from llm_selenium_agent import BaseSeleniumChrome
from selenium.webdriver.common.by import By

class MyAutomation(BaseSeleniumChrome):
    def __init__(self):
        super().__init__()
        self.url = "https://www.example.com"  # Website to automate
    
    def login(self):
        # Example login implementation
        username_field = self.driver.find_element(By.ID, "username")
        username_field.send_keys("your_username")
        
        password_field = self.driver.find_element(By.ID, "password")
        password_field.send_keys("your_password")
        
        login_button = self.driver.find_element(By.ID, "login-button")
        login_button.click()

# Run the automation
if __name__ == "__main__":
    automation = MyAutomation()
    automation.main()
```

## License

MIT

## Author

Lunde Chen - lundechen@shu.edu.cn (Shanghai University)