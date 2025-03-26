import logging


def make_simple_logger(name: str, add_file_handler: bool = False) -> logging.Logger:
    """Create a simple logger with a stream handler.

    Args:
        name (str): Name of the logger
        add_file_handler (bool, optional): Add a file handler. Defaults to False.

    Returns:
        logging.Logger: Logger object
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        if add_file_handler:
            fh = logging.FileHandler(f"{name}.log")
            fh.setLevel(logging.INFO)
            fh.setFormatter(formatter)
            logger.addHandler(fh)

    return logger
