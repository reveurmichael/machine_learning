import streamlit as st
import os
from llm_selenium_agent.streamlit_tools import configure_page_navigation
from llm_selenium_agent.config import *
from llm_selenium_agent.network import *

class App:
    def __init__(self):
        self.setup_page_config()
        configure_page_navigation("app")
        self.main()

    def setup_page_config(self):
        # Configure the page with a wide layout and custom title/icon
        st.set_page_config(
            page_title="Selenium WebDriver Setup",
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="expanded",
        )

        # Load and inject CSS
        common_css_path = os.path.join(os.path.dirname(__file__), "css", "common.css")
        with open(common_css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

        css_path = os.path.join(os.path.dirname(__file__), "css", "app.css")
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

        # Main header
        st.markdown(
            '<h1 class="common-title">🤖 Selenium WebDriver Setup Tool</h1>',
            unsafe_allow_html=True,
        )

    def display_welcome_section(self):
        st.markdown("## Welcome to the Selenium Setup Tool for students!")
        st.markdown("""
        This tool will help you get started with Selenium WebDriver for browser automation.
        Follow the steps below to set up your environment and start creating your own automation scripts.
        """)
        
        # Add a simple info box with key points
        st.info("""
        **What is Selenium?** Selenium WebDriver is a tool for automated browser testing and web scraping.
        It allows you to control web browsers programmatically and simulate user interactions.
        """)

    def display_steps_section(self):
        st.markdown("## 📋 Getting Started in 3 Steps")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### Step 1: Install Drivers")
            st.markdown("""
            First, install the WebDrivers for Chrome and Firefox using the buttons in the 'WebDriver Installation' section below.
            
            WebDrivers are the components that allow Selenium to communicate with and control browsers.
            """)
        
        with col2:
            st.markdown("### Step 2: Configure Settings")
            st.markdown("""
            Adjust settings like 'headless mode' which runs browsers without displaying windows.
            
            This is useful for running scripts on servers or when you don't need to see the browser window.
            """)
        
        with col3:
            st.markdown("### Step 3: Test Connection")
            st.markdown("""
            Verify that your WebDrivers can communicate with browsers by running the test connections.
            
            If all tests pass, you're ready to start creating your own automation scripts!
            """)

    def display_driver_section(self):
        st.markdown(
            '<div class="section-header">🔧 WebDriver Installation</div>',
            unsafe_allow_html=True,
        )
        
        chrome_driver_path = get_chrome_driver_path()
        firefox_driver_path = get_firefox_driver_path()
        
        # Chrome Driver status
        st.markdown("### Chrome WebDriver")
        if chrome_driver_path and os.path.exists(chrome_driver_path):
            st.success(f"✅ ChromeDriver is installed at: {chrome_driver_path}")
        else:
            st.warning("⚠️ ChromeDriver is not installed")
        
        if st.button("🔄 Install Chrome WebDriver"):
            with st.spinner("🔧 Installing Chrome WebDriver..."):
                success, message, path = install_chrome_driver()
                if success == 1:
                    st.success(f"✅ {message}")
                    st.info(f"📍 Location: {path}")
                else:
                    st.error(f"❌ {message}")
                    st.error("This might be due to network issues.")
        
        # Firefox Driver status
        st.markdown("### Firefox WebDriver")
        if firefox_driver_path and os.path.exists(firefox_driver_path):
            st.success(f"✅ Firefox WebDriver is installed at: {firefox_driver_path}")
        else:
            st.warning("⚠️ Firefox WebDriver is not installed")
            
        if st.button("🔄 Install Firefox WebDriver"):
            with st.spinner("🔧 Installing Firefox WebDriver..."):
                success, message, path = install_firefox_driver()
                if success == 1:
                    st.success(f"✅ {message}")
                    st.info(f"📍 Location: {path}")
                else:
                    st.error(f"❌ {message}")
                    st.error("This might be due to network issues.")

    def display_settings_section(self):
        st.markdown(
            '<div class="section-header">⚙️ Basic Settings</div>',
            unsafe_allow_html=True,
        )
        
        config = load_configuration()
        
        # Ensure selenium section exists
        if "selenium" not in config:
            config["selenium"] = {}
            
        # Headless Mode
        headless = st.checkbox(
            "Run browsers in headless mode",
            value=str(config.get("selenium", {}).get("headless", False)).lower() == "true",
            help="When enabled, browser windows won't be visible during automation."
        )
        config["selenium"]["headless"] = str(headless).lower()
        update_configuration(config)
        
        # Chrome options
        if "chrome_options" not in config:
            config["chrome_options"] = {}
            
        ignore_cert_errors = st.checkbox(
            "Ignore certificate errors in Chrome",
            value=str(config.get("chrome_options", {}).get("ignore-certificate-errors", "true")).lower() == "true",
            help="Ignore SSL certificate errors in Chrome."
        )
        config["chrome_options"]["ignore-certificate-errors"] = str(ignore_cert_errors).lower()
        update_configuration(config)

    def display_test_section(self):
        st.markdown(
            '<div class="section-header">🧪 Test Connection</div>',
            unsafe_allow_html=True,
        )
        
        # Add info message for students in China
        st.info("""
        **Note for students in China:** If you cannot access Google, please use the Baidu test options below.
        Testing with Baidu.com will verify that your WebDrivers are working correctly.
        """)
        
        tab1, tab2 = st.tabs(["Google Tests", "Baidu Tests (for China)"])
        
        # Google Tests Tab
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Test Chrome with Google")
                if st.button("🔍 Test Chrome with Google.com"):
                    with st.spinner("🔄 Testing connection to Google.com with Chrome..."):
                        success = verify_google_chrome_accessibility()
                        if success:
                            st.success("✅ Chrome successfully accessed Google.com")
                        else:
                            st.error("❌ Failed to access Google.com with Chrome")
                            st.info("Make sure ChromeDriver is installed and Chrome browser is available. If you're in China, try the Baidu test instead.")
            
            with col2:
                st.markdown("### Test Firefox with Google")
                if st.button("🔍 Test Firefox with Google.com"):
                    with st.spinner("🔄 Testing connection to Google.com with Firefox..."):
                        success = verify_google_firefox_accessibility()
                        if success:
                            st.success("✅ Firefox successfully accessed Google.com")
                        else:
                            st.error("❌ Failed to access Google.com with Firefox")
                            st.info("Make sure GeckoDriver is installed and Firefox browser is available. If you're in China, try the Baidu test instead.")
        
        # Baidu Tests Tab
        with tab2:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Test Chrome with Baidu")
                if st.button("🔍 Test Chrome with Baidu.com"):
                    with st.spinner("🔄 Testing connection to Baidu.com with Chrome..."):
                        success = verify_baidu_chrome_accessibility()
                        if success:
                            st.success("✅ Chrome successfully accessed Baidu.com")
                        else:
                            st.error("❌ Failed to access Baidu.com with Chrome")
                            st.info("Make sure ChromeDriver is installed and Chrome browser is available.")
            
            with col2:
                st.markdown("### Test Firefox with Baidu")
                if st.button("🔍 Test Firefox with Baidu.com"):
                    with st.spinner("🔄 Testing connection to Baidu.com with Firefox..."):
                        success = verify_baidu_firefox_accessibility()
                        if success:
                            st.success("✅ Firefox successfully accessed Baidu.com")
                        else:
                            st.error("❌ Failed to access Baidu.com with Firefox")
                            st.info("Make sure GeckoDriver is installed and Firefox browser is available.")

    def display_code_example(self):
        st.markdown(
            '<div class="section-header">💻 Example Code</div>',
            unsafe_allow_html=True,
        )
        
        tab1, tab2 = st.tabs(["Google Search Example", "Baidu Search Example"])
        
        with tab1:
            st.markdown("""
            ### Google Search Example
            
            Copy this code to automate a Google search:
            """)
            
            code_google = '''
# Google Search automation example
from llm_selenium_agent import BaseSeleniumChrome
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

class GoogleSearch(BaseSeleniumChrome):
    def __init__(self):
        super().__init__()
        self.url = "https://www.google.com"  # Website to automate
    
    def login(self):
        # Google doesn't need login for this example
        pass
    
    def run_search(self, search_term):
        # Wait for search box to be present
        search_box = WebDriverWait(self.driver, 10).until(
            EC.presence_of_element_located((By.NAME, "q"))
        )
        
        # Type the search term
        search_box.send_keys(search_term)
        
        # Submit the search
        search_box.submit()
        
        # Wait for results
        WebDriverWait(self.driver, 10).until(
            EC.presence_of_element_located((By.ID, "search"))
        )

# Run the automation
if __name__ == "__main__":
    automation = GoogleSearch()
    
    try:
        automation.prepare_environment()
        automation.navigate_to_url()
        automation.run_search("Selenium Python tutorial")
        
        # Wait to see results
        import time
        time.sleep(5)
    finally:
        automation.terminate_webdriver()
'''
            
            st.code(code_google, language="python")
        
        with tab2:
            st.markdown("""
            ### Baidu Search Example (for China)
            
            Copy this code to automate a Baidu search:
            """)
            
            code_baidu = '''
# Baidu Search automation example (for China)
from llm_selenium_agent import BaseSeleniumChrome
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

class BaiduSearch(BaseSeleniumChrome):
    def __init__(self):
        super().__init__()
        self.url = "https://www.baidu.com"  # Baidu URL
    
    def login(self):
        # Baidu doesn't need login for this example
        pass
    
    def run_search(self, search_term):
        # Wait for search box to be present
        search_box = WebDriverWait(self.driver, 10).until(
            EC.presence_of_element_located((By.ID, "kw"))
        )
        
        # Type the search term
        search_box.send_keys(search_term)
        
        # Submit the search
        search_button = WebDriverWait(self.driver, 10).until(
            EC.element_to_be_clickable((By.ID, "su"))
        )
        search_button.click()
        
        # Wait for results to load
        WebDriverWait(self.driver, 10).until(
            EC.presence_of_element_located((By.ID, "content_left"))
        )

# Run the automation
if __name__ == "__main__":
    automation = BaiduSearch()
    
    try:
        automation.prepare_environment()
        automation.navigate_to_url()
        automation.run_search("Selenium Python 教程")
        
        # Wait to see results
        import time
        time.sleep(5)
    finally:
        automation.terminate_webdriver()
'''
            
            st.code(code_baidu, language="python")
            
        st.markdown("""
        ### How to Run
        
        1. Save the code to a file (e.g., `search_automation.py`)
        2. Run it with Python: `python search_automation.py`
        3. Watch as the browser opens, enters your search term, and displays results!
        """)

    def main(self):
        # Display all sections in the desired order
        self.display_welcome_section()
        self.display_steps_section()
        self.display_driver_section()
        self.display_settings_section()
        self.display_test_section()
        self.display_code_example()


if __name__ == "__main__":
    App()
