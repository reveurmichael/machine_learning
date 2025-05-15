from .env import get_environment_variable
from .logger import logger, log_function_call
import requests
import base64
import time
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from typing import Optional
from .decorator import *


@log_function_call
def get_twocaptcha_api_key() -> str:
    """Retrieve the 2Captcha API key from environment variables."""
    return get_environment_variable("TWOCAPTCHA_API_KEY")


@log_function_call
def get_twocaptcha_balance() -> tuple[bool, str]:
    """Check the balance of the 2Captcha account."""
    response = requests.get(
        f"https://2captcha.com/res.php?key={get_twocaptcha_api_key()}&action=getbalance"
    )
    logger.info(f"2Captcha Response Code: {response.status_code}")
    logger.info(f"2Captcha Response Text: {response.text}")
    success = response.status_code == 200
    if success:
        balance = response.text
        info = f"API key is valid. Your balance is: {balance} credits."
        logger.info(info)
    else:
        info = (
            f"Failed to connect to 2Captcha API. "
            f"Maybe you need to recharge your balance? Response code: {response.status_code}"
        )
        logger.warning(info)
    return success, info


@log_function_call
def solve_captcha_image(driver, img_element_locator: tuple) -> Optional[str]:
    """
    Solve CAPTCHA by sending the image to 2Captcha and retrieving the solution.

    Args:
        driver: Selenium WebDriver instance.
        img_element_locator: Locator tuple for the CAPTCHA image element.

    Returns:
        The CAPTCHA solution as a string if successful, else None.
    """
    img_element = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located(img_element_locator)
    )

    # Save the image element directly as captcha.jpg
    img_element.screenshot('captcha.jpg')  # TODO

    # Encode the CAPTCHA image in base64
    with open("captcha.jpg", "rb") as img_file:
        captcha_image = img_file.read()
        captcha_base64 = base64.b64encode(captcha_image).decode("utf-8")

    # Send the CAPTCHA image to 2Captcha
    captcha_response = requests.post(
        "http://2captcha.com/in.php",
        data={
            "key": get_twocaptcha_api_key(),
            "method": "base64",
            "body": captcha_base64,
            "json": 1,
        },
    ).json()

    if captcha_response.get("status") == 1:
        captcha_id = captcha_response.get("request")
        time.sleep(20)  # Wait for CAPTCHA to be solved
        result_response = requests.get(
            f"http://2captcha.com/res.php?key={get_twocaptcha_api_key()}&action=get&id={captcha_id}&json=1"
        ).json()

        if result_response.get("status") == 1:
            solution = result_response.get("request")
            logger.info(f"2Captcha Solution: {solution}")
            return solution
        else:
            logger.error(f"Error retrieving CAPTCHA solution: {result_response.get('request')}")
            return None
    else:
        logger.error(f"Error sending CAPTCHA to 2Captcha: {captcha_response.get('request')}")
        return None

