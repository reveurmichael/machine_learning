from .base_selenium import *
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.firefox.firefox_profile import FirefoxProfile


class BaseSeleniumFirefox(BaseSelenium):
    def __init__(self):
        super().__init__()
        self.browser_name = "Firefox"

    @log_function_call
    def setup_options(self):
        self.options = FirefoxOptions()
        if self.headless_mode:
            self.options.add_argument("--headless")
        else:
            self.options.set_preference("browser.startup.homepage", "about:blank")
            self.options.set_preference("browser.startup.page", 0)
            self.options.set_preference(
                "dom.disable_open_during_load", False
            )  

        firefox_binary = get_firefox_binary_location()
        if firefox_binary:
            self.options.binary_location = firefox_binary
        else:
            logger.warning("Firefox binary location not found in configuration.")

        self.options.set_preference("network.proxy.allow_bypass_local", True)
        self.options.set_preference("security.ssl.enable_ocsp_stapling", False)
        self.options.set_preference("security.ssl.errorReporting.automatic", False)
        self.options.set_preference("security.ssl.errorReporting.enabled", False)
        self.options.set_preference("security.ssl.enable_ocsp_stapling", False)
        self.options.set_preference(
            "security.mixed_content.block_active_content", False
        )  # Allow mixed content
        self.options.set_preference(
            "security.mixed_content.block_display_content", False
        )  # Allow display content

        self.profile = FirefoxProfile()
        self.profile.set_preference("browser.download.folderList", 2)  # Use custom download directory
        self.profile.set_preference("browser.download.dir", self.downloads_dir)
        self.profile.set_preference("browser.helperApps.neverAsk.saveToDisk", "application/octet-stream,application/pdf,image/png,image/jpeg")  # Add MIME types as needed
        self.profile.set_preference("browser.download.manager.showWhenStarting", False)
        self.profile.set_preference("browser.download.manager.useWindow", False)
        self.profile.set_preference("browser.download.manager.focusWhenStarting", False)
        self.profile.set_preference("browser.download.manager.alertOnEXEOpen", False)
        self.profile.set_preference("browser.download.manager.showAlertOnComplete", False)
        self.profile.set_preference("browser.download.manager.useWindow", False)
        self.profile.set_preference(
            "pdfjs.disabled", True
        )

        self.options.profile = self.profile

    @log_function_call
    def initialize_webdriver(self):
        self.setup_options()
        self.driver = webdriver.Firefox(
            service=get_firefox_service_instance(), 
            options=self.options, 
        )
        return self.driver
