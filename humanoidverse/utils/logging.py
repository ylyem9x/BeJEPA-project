import logging
import os
import sys
from contextlib import contextmanager

from loguru import logger


class HydraLoggerBridge(logging.Handler):
    def emit(self, record):
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        (frame, depth) = (logging.currentframe(), 2)
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1
        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


class LoguruStream:
    def write(self, message):
        if message.strip():
            logger.info(message.strip())

    def flush(self):
        pass


@contextmanager
def capture_stdout_to_loguru():
    logger.remove()
    logger.add(sys.stdout, level="INFO")
    loguru_stream = LoguruStream()
    old_stdout = sys.stdout
    sys.stdout = loguru_stream
    try:
        yield
    finally:
        sys.stdout = old_stdout
        logger.remove()
        console_log_level = os.environ.get("LOGURU_LEVEL", "INFO").upper()
        logger.add(sys.stdout, level=console_log_level, colorize=True)
