import time

def log_function_call(func):
    """Decorator to log function calls."""

    def wrapper(*args, **kwargs):
        from .logger import logger  # Lazy import to avoid circular dependency
        logger.debug(
            f"Called function: {func.__name__} with args: {args} and kwargs: {kwargs}"
        )
        return func(*args, **kwargs)

    return wrapper


def retry(func, retries=3, delay=3):
    """Decorator to retry a function."""
    def wrapper(*args, **kwargs):
        for attempt in range(retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt < retries - 1:  # If not the last attempt
                    time.sleep(delay)  # Wait before retrying
                else:
                    raise e  # Re-raise the last exception
    return wrapper

