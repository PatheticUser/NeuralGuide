import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Centralized settings, constants
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# UI Settings
APP_TITLE = "AgentLens"
APP_TAGLINE = "AI-Powered LLM Discovery Assistant for Agentic AI Workflows"
APP_DESCRIPTION = "An end-to-end engineering project designed to help developers and researchers navigate the rapidly evolving landscape of Large Language Models (LLMs)."
