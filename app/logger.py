import sys

from loguru import logger

def setup_logger():
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
    )
    logger.add(
        "logs/exceptions.json",
        level="ERROR",
        serialize=True,
        rotation="10 MB",
        catch=True,
    )
    return logger
