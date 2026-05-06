from dotenv import load_dotenv

# Load `.env` from the dashboard project root before any module touches
# os.environ (e.g. the Anthropic SDK on import).
load_dotenv()

from esp_dashboard.esp_dashboard import app  # noqa: E402

__all__ = ["app"]
