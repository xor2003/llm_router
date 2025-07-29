import sys

from loguru import logger

from app.pii import PIIScrubber


def setup_logger():
    """
    Configures the application logger with PII scrubbing.
    """
    pii_scrubber = PIIScrubber()

    def pii_patcher(record):
        """
        A loguru patcher that scrubs PII from log records before they are processed.
        It modifies the record in-place.
        """
        # Scrub the main log message
        record["message"], _ = pii_scrubber._scrub_text(record["message"])

        # Scrub any data passed in the 'extra' dict
        for key, value in record["extra"].items():
            if isinstance(value, str):
                record["extra"][key], _ = pii_scrubber._scrub_text(value)
            elif isinstance(value, dict):
                scrubbed_payload, _ = pii_scrubber.scrub(value)
                record["extra"][key] = scrubbed_payload

        # Scrub the exception details if present
        if record["exception"]:
            exc_type, exc_value, exc_traceback = record["exception"]
            try:
                # Recreate the exception with scrubbed arguments to avoid logging sensitive data
                scrubbed_args = []
                for arg in exc_value.args:
                    # We can only reliably scrub string arguments
                    if isinstance(arg, str):
                        scrubbed_args.append(pii_scrubber._scrub_text(arg)[0])
                    else:
                        scrubbed_args.append(arg)

                new_exc_value = exc_type(*scrubbed_args)
                record["exception"] = (exc_type, new_exc_value, exc_traceback)
            except Exception:
                # If we fail to recreate the exception, we don't want to crash the logger.
                # We'll proceed with the original exception. Some PII might leak in this edge case,
                # but it's better than losing the log entry entirely.
                pass

    logger.remove()
    logger.patch(pii_patcher)

    # Console logger
    logger.add(
        sys.stderr,
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
    )

    # File logger for exceptions
    logger.add(
        "logs/exceptions.json",
        level="ERROR",
        serialize=True,
        rotation="10 MB",
        catch=True,  # Catches unhandled exceptions
    )
    return logger
