import logging
import os
import warnings

from dotenv import load_dotenv

from esp_data.paths import AnyPath

load_dotenv()

warnings.filterwarnings("ignore", "Your application has authenticated using end user credentials")
CONFIG = {
    "DEBUG": os.getenv("DEBUG", "False") == "True",
}

# make root logger
logger = logging.getLogger("esp_data")

# Set up a handler for this logger only
if CONFIG["DEBUG"]:
    level = logging.DEBUG
else:
    level = logging.INFO

logger.setLevel(level)

# Only add a handler if one doesn't exist yet
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)

__all__ = ["AnyPath", "CONFIG"]
