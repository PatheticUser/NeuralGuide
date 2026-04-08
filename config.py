import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

# Load environment variables from .env file
load_dotenv()

class Settings(BaseSettings):
    OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    
    # We override OpenAI environment vars so openai-agents SDK uses local Ollama
    OPENAI_BASE_URL: str = f"{OLLAMA_HOST}/v1"
    OPENAI_API_KEY: str = "ollama"
    
    # UI Constants
    APP_TITLE: str = "NeuralGuide"
    APP_TAGLINE: str = "AI-Powered LLM Discovery Assistant"
    APP_DESCRIPTION: str = "An end-to-end engineering project designed to help developers and researchers navigate the rapidly evolving landscape of Large Language Models (LLMs)."

settings = Settings()

# Set env vars explicitly so openai sdk picks them up instantly
os.environ["OPENAI_BASE_URL"] = settings.OPENAI_BASE_URL
os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY

# Alias for backward compatibility
OPENAI_API_KEY = settings.OPENAI_API_KEY
OLLAMA_HOST = settings.OLLAMA_HOST
APP_TITLE = settings.APP_TITLE
APP_TAGLINE = settings.APP_TAGLINE
APP_DESCRIPTION = settings.APP_DESCRIPTION
