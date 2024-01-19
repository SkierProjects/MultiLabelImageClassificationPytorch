import logging
import logging.handlers
import os

class LoggerFactory:
    DEFAULT_LOG_LEVEL = logging.INFO
    LOG_FILE_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
    LOG_FILE_BACKUP_COUNT = 5  # Keep 5 backup files
    LONG_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    SHORT_LOG_FORMAT = "%(levelname)s: %(message)s"
    DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

    @staticmethod
    def setup_logging(loggername, config, log_file=None, level=None):
        """
        Set up logging configuration for a logger with the specified name.

        Parameters:
            logger_name (str): The name of the logger to set up.
            log_file (str): The path to the log file. If None, logs to stdout.
            level (int): The logging level. If None, defaults to the level specified in config.
            config (module): The configuration module with a 'log_level' attribute.

        Returns:
            logging.Logger: Configured logger instance.
        """
        if level is None:
            level = getattr(logging, config.logs_level, LoggerFactory.DEFAULT_LOG_LEVEL)
        
        # Since we are setting up handlers individually, we don't use basicConfig
        logger = logging.getLogger(loggername)
        logger.setLevel(level)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(LoggerFactory.SHORT_LOG_FORMAT))
        logger.addHandler(console_handler)

        if log_file is not None:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.handlers.RotatingFileHandler(
                log_file, maxBytes=LoggerFactory.LOG_FILE_MAX_BYTES, backupCount=LoggerFactory.LOG_FILE_BACKUP_COUNT)
            file_handler.setFormatter(logging.Formatter(LoggerFactory.LONG_LOG_FORMAT, LoggerFactory.DATE_FORMAT))
            logger.addHandler(file_handler)

        return logger

    @staticmethod
    def get_logger(name):
        """
        Get a logger with the specified name.

        Parameters:
            name (str): The name of the logger to retrieve.

        Returns:
            logging.Logger: The logger instance with the given name.
        """
        return logging.getLogger(name)