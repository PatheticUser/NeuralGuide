import ollama
from config import OLLAMA_HOST

def get_local_models() -> list[dict]:
    """
    Fetches the list of locally available models from the Ollama API using the ollama package.
    Returns a list of dictionaries with model metadata.
    """
    try:
        # In the official ollama python client, you can construct a client with host
        client = ollama.Client(host=OLLAMA_HOST)
        response = client.list()
        
        models = []
        for model in response.get("models", []):
            models.append({
                "Model Name": model.get("name", "Unknown"),
                "Description": "Local model served via Ollama.",
                "Parameters": model.get("details", {}).get("parameter_size", "Unknown"),
                "Key Features": "Privacy, Offline capability",
                "Tool/Function Calling": "Depends on specific model"
            })
        return models
    except Exception as e:
        print(f"Error connecting to Ollama: {e}")
        return []

def is_ollama_running() -> bool:
    """
    Checks if the Ollama service is running and accessible.
    """
    try:
        client = ollama.Client(host=OLLAMA_HOST)
        client.list()
        return True
    except Exception:
        return False
