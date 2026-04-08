from rich.console import Console
from theme import Icons, Colors
from openai import OpenAI
import config # ensures env vars are loaded

console = Console()

def is_openai_configured() -> bool:
    try:
        client = OpenAI()
        client.models.list()
        return True
    except Exception:
        return False

def get_openai_status() -> tuple[bool, str]:
    """Returns connectivity status and a nice string with an icon."""
    if is_openai_configured():
        return True, f"[bold {Colors.SUCCESS}]{Icons.ONLINE}[/] [{Colors.SUCCESS}]Local Ollama Connected[/]"
    return False, f"[bold {Colors.ERROR}]{Icons.OFFLINE}[/] [{Colors.ERROR}]Local Ollama Offline[/]"

def get_local_models() -> list[dict]:
    try:
        client = OpenAI()
        response = client.models.list()
        models = []
        for model in response.data:
            models.append({
                "Model Name": model.id,
                "Description": "Local model served via Ollama.",
                "Parameters": "Unknown",
                "Key Features": "Local Execution via OpenAI SDK",
                "Tool/Function Calling": "Determined by architecture"
            })
        return models
    except Exception:
        return []
