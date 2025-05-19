from .base_selenium import *
from selenium.webdriver.chrome.options import Options as ChromeOptions


class BaseSeleniumChrome(BaseSelenium):
    @log_function_call
    def __init__(self):
        super().__init__()
        self.browser_name = "Chrome"

    @log_function_call
    def setup_options(self):
        self.options = ChromeOptions()
        if self.headless_mode:
            self.options.add_argument("--headless")
        self.options.add_argument("--window-size=1920,1080")
        
        # Check if chrome_options exists in config to avoid KeyError
        chrome_options = self.config.get("chrome_options", {})
        if chrome_options and chrome_options.get("ignore-certificate-errors") == "true":
            self.options.add_argument("--ignore-certificate-errors")
            
        self.options.add_argument("--ignore-ssl-errors=yes")
        self.options.add_argument("--allow-running-insecure-content")
        self.options.add_argument("--allow-insecure-localhost")
        self.options.add_experimental_option(
            "prefs",
            {
                "download.default_directory": self.downloads_dir,
                "download.prompt_for_download": False,
                "download.directory_upgrade": True,
                "safebrowsing.enabled": False,
                "safebrowsing.disable_download_protection": True,
            },
        )

    @log_function_call
    def initialize_webdriver(self):
        self.setup_options()
        self.driver = webdriver.Chrome(
            service=get_chrome_service_instance(),
            options=self.options,
        )
        return self.driver
