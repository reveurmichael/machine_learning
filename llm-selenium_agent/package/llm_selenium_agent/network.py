import os
import yaml
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import WebDriverException
from .config import *
from .logger import logger
from .decorator import log_function_call
from webdriver_manager.firefox import GeckoDriverManager
from selenium.webdriver.firefox.service import Service as FirefoxService
import shutil

@log_function_call
def get_firefox_service_instance():
    """Get the FirefoxService instance."""
    firefox_driver_path = get_firefox_driver_path()

    if firefox_driver_path and os.path.exists(firefox_driver_path):
        logger.info("Local FirefoxDriver found.")
        service = FirefoxService(firefox_driver_path)
    else:
        logger.warning("Local FirefoxDriver not found. Installing...")
        service = FirefoxService(GeckoDriverManager().install())
        logger.info("FirefoxDriver installation completed.")
    return service


@log_function_call
def install_firefox_driver():
    """
    Install GeckoDriver (FirefoxDriver) and update the config.yml file with the driver path.
    """
    config = load_configuration()
    service = get_firefox_service_instance()

    firefox_driver_path = service.path

    config.setdefault("selenium", {})["firefox_driver_path"] = firefox_driver_path

    update_configuration(config)

    logger.info(
        f"FirefoxDriver installed and config updated. Path: {firefox_driver_path}"
    )
    firefox_driver_path = get_firefox_driver_path()
    if firefox_driver_path and os.path.exists(firefox_driver_path):
        return 1, f"Firefox Driver installed successfully.", firefox_driver_path
    else:
        return 2, f"Firefox Driver installation failed.", firefox_driver_path


@log_function_call
def get_chrome_service_instance():
    """Get the ChromeService instance."""
    chrome_driver_path = get_chrome_driver_path()

    if chrome_driver_path and os.path.exists(chrome_driver_path):
        logger.info("Local ChromeDriver found.")
        service = ChromeService(chrome_driver_path)
    else:
        logger.warning("Local ChromeDriver not found. Installing...")
        service = ChromeService(ChromeDriverManager().install())
        logger.info("ChromeDriver installation completed.")
    return service


def install_chrome_and_firefox_drivers():
    install_chrome_driver()
    install_firefox_driver()
    chrome_driver_path = get_chrome_driver_path()
    firefox_driver_path = get_firefox_driver_path()

    if chrome_driver_path and os.path.exists(chrome_driver_path) and firefox_driver_path and os.path.exists(firefox_driver_path):
        return 1, f"Chrome and Firefox Drivers installed successfully.", chrome_driver_path, firefox_driver_path
    elif (not chrome_driver_path or not os.path.exists(chrome_driver_path)) and (not firefox_driver_path or not os.path.exists(firefox_driver_path)):
        return 2, f"Chrome and Firefox Drivers installation failed.", chrome_driver_path, firefox_driver_path
    elif not chrome_driver_path or not os.path.exists(chrome_driver_path):
        return 3, f"Firefox Driver installed successfully but Chrome Driver installation failed.", chrome_driver_path, firefox_driver_path
    elif not firefox_driver_path or not os.path.exists(firefox_driver_path):
        return 4, f"Chrome Driver installed successfully but Firefox Driver installation failed.", chrome_driver_path, firefox_driver_path


@log_function_call
def remove_all_drivers():
    """Remove all the drivers in the `~/.wdm/drivers/` directory recursively. This is useful if you have corrupted or out-of-date drivers in your systems. """
    drivers_directory = os.path.expanduser("~/.wdm/drivers/")
    if os.path.exists(drivers_directory):
        shutil.rmtree(drivers_directory)
    else:
        logger.info(f"Directory {drivers_directory} does not exist.")


@log_function_call
def install_chrome_driver():
    """
    Install ChromeDriver and update the config.yml file with the driver path.
    """
    config = load_configuration()
    service = get_chrome_service_instance()

    chrome_driver_path = service.path

    config.setdefault("selenium", {})["chrome_driver_path"] = chrome_driver_path

    update_configuration(config)

    logger.info(
        f"ChromeDriver installed and config updated. Path: {chrome_driver_path}"
    )
    chrome_driver_path = get_chrome_driver_path()
    if chrome_driver_path and os.path.exists(chrome_driver_path):
        return 1, f"Chrome Driver installed successfully.", chrome_driver_path
    else:
        return 2, f"Chrome Driver installation failed.", chrome_driver_path


def verify_google_chrome_accessibility():
    try:
        driver = webdriver.Chrome(service=get_chrome_service_instance())
        driver.get("https://www.google.com")
        if "Google" in driver.title:
            logger.info("Google is accessible.")
            time.sleep(3)
            return True
        else:
            logger.info("Google is not accessible.")
            return False
    except WebDriverException as e:
        logger.error(f"WebDriverException: {e}")
        return False
    finally:
        driver.quit()


def verify_baidu_chrome_accessibility():
    try:
        driver = webdriver.Chrome(service=get_chrome_service_instance())
        driver.get("https://www.baidu.com")
        time.sleep(3)
        if "百度" in driver.title:
            logger.info("Baidu is accessible with Chrome.")
            return True
        else:
            logger.info("Baidu is not accessible with Chrome.")
            return False
    except WebDriverException as e:
        logger.error(f"WebDriverException: {e}")
        return False
    finally:
        driver.quit()


def verify_baidu_firefox_accessibility():
    try:
        driver = webdriver.Firefox(service=get_firefox_service_instance())
        driver.get("https://www.baidu.com")
        time.sleep(3)
        if "百度" in driver.title:
            logger.info("Baidu is accessible.")
            return True
        else:
            logger.info("Baidu is not accessible.")
            return False
    except WebDriverException as e:
        logger.error(f"WebDriverException: {e}")
        return False
    finally:
        driver.quit()


def verify_google_firefox_accessibility():
    try:
        driver = webdriver.Firefox(service=get_firefox_service_instance())
        driver.get("https://www.google.com")
        if "Google" in driver.title:
            logger.info("Google is accessible.")
            time.sleep(3)
            return True
        else:
            logger.info("Google is not accessible.")
            return False
    except WebDriverException as e:
        logger.error(f"WebDriverException: {e}")
        return False
    finally:
        driver.quit()
