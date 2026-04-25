import logging
from enum import StrEnum

# Format for DEBUG logs (includes level, message, file path, function, and line number)
LOG_FORMAT_DEBUG = "%(levelname)s:%(message)s:%(pathname)s:%(funcName)s:%(lineno)d"


# Enum for log levels, using string values for easy comparison
class logLevels(StrEnum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


def configure_logging(log_level: str = logLevels.ERROR):
    """
    Configure Python logging based on the provided log level.

    Args:
        log_level (str): Desired log level. Should be one of 'DEBUG', 'INFO', 'WARNING', 'ERROR'.
                         Defaults to 'ERROR' if invalid input is provided.

    Behavior:
        - If log_level is DEBUG, uses a detailed debug format.
        - For other levels, uses default logging format.
        - If log_level is invalid, defaults to ERROR level.
    """
    # Convert input to uppercase string
    log_level = str(log_level).upper()
    # List of valid log levels
    log_levels = [level.value for level in logLevels]

    # Default to ERROR if invalid log_level is given
    if log_level not in log_levels: 
        logging.basicConfig(level=logging.ERROR)
        return
    
    # Special format for DEBUG
    if log_level == logLevels.DEBUG:
        logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT_DEBUG)
        return
    
    # Configure logging for other valid levels
    logging.basicConfig(level=log_level)
