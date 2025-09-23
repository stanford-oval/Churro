from loguru import logger
from rich.logging import RichHandler


# Remove the default loguru handler
logger.remove()

# Add a new handler using RichHandler for console output
logger.add(
    RichHandler(markup=True, show_time=False),  # Enable rich markup for colored output
    level="INFO",  # Set the logging level
    format="{message}",
    backtrace=True,  # Include the backtrace in the log
    diagnose=True,  # Include diagnostic information in the log
)

# Add another handler for saving debug logs to a file
logger.add(
    "debug_logs.log",  # File path for the log file
    level="DEBUG",  # Set the logging level to DEBUG
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name} | {message}",  # Log format
    rotation="5 MB",  # Rotate the log file when it reaches 5 MB
    retention=2,  # Keep the last 2 log files
    backtrace=True,  # Include the backtrace in the log
    diagnose=True,  # Include diagnostic information in the log
    enqueue=True,
)
