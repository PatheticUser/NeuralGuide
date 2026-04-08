import json
from pydantic import BaseModel, Field
from config import OPENAI_API_KEY
from tavily_utils import search_tavily
from rich.console import Console
from agents import Agent, Runner

console = Console()

class ModelInfo(BaseModel):
    model_name: str = Field(description="Official name and version")
    description: str = Field(description="Summary of strengths for this specific task")
    parameters: str = Field(description="Size in billions (e.g., 8B, 70B, etc. or N/A)")
    key_features: str = Field(description="Key strengths, benchmarks, or reasoning capabilities")
    tool_calling: str = Field(description="Support for parallel tool or function execution")

class ModelsResponse(BaseModel):
    models: list[ModelInfo]

def get_agentic_models_from_cloud(query: str) -> list[dict]:
    """
    Performs deep research using Tavily, then compiles a top model recommendation list
    using openai-agents SDK.
    """
    # 1. Start Web Research
    research_context = search_tavily(query)

    # 2. Compile system prompt with research data
    system_prompt = (
        "You are 'NeuralGuide', an AI Research & Discovery Agent. "
        "Your task is to perform an exhaustive evaluation of models for the user's specific query. "
        "Use the provided Research Context for latest benchmarks and pricing. "
        "CRITICAL REQUIREMENT: You MUST recommend EXACTLY 7 distinct models. NOT 1, NOT 3, but EXACTLY 7. "
        "Provide a diverse range: include at least 2 frontier models, 3 open-source models, and 2 task-specific niche models. "
        "\n\n--- RESEARCH CONTEXT ---\n"
        f"{research_context}\n"
        "--- END RESEARCH CONTEXT ---\n\n"
        "Output MUST be a raw JSON object matching this schema exactly. "
        "Your JSON MUST contain exactly 7 items in the 'models' array:\n"
        "{\n"
        "  \"models\": [\n"
        "    { \"model_name\": \"...\", \"description\": \"...\", \"parameters\": \"...\", \"key_features\": \"...\", \"tool_calling\": \"...\" }\n"
        "  ]\n"
        "}\n\nDo not output markdown code blocks. Just valid JSON."
    )

    try:
        from openai import OpenAI
        client = OpenAI()
        models_response = client.models.list()
        available_models = [m.id for m in models_response.data]

        # Engine choice strategy
        engine_options = ["qwen3.5:cloud", "deepseek-v3.1:671b-cloud", "nemotron-3-super:cloud"]
        engine_model = available_models[0] if available_models else "gpt-4o"
        for opt in engine_options:
            if opt in available_models:
                engine_model = opt
                break

        agent = Agent(name="NeuralGuide", instructions=system_prompt, model=engine_model)
        result = Runner.run_sync(agent, f"Discover top 7 models for: {query}")
        raw_content = result.final_output
        
        # Parse JSON and strip markdown backticks if present
        if raw_content.startswith("```json"):
            raw_content = raw_content.split("```json")[1].split("```")[0].strip()
        elif raw_content.startswith("```"):
            raw_content = raw_content.split("```")[1].split("```")[0].strip()
            
        data = json.loads(raw_content)
        
        # Format the result list for UI component
        results = []
        for m in data.get("models", []):
            results.append({
                "Model Name": m.get("model_name", "Unknown"),
                "Description": m.get("description", "Not provided"),
                "Parameters": m.get("parameters", "N/A"),
                "Key Features": m.get("key_features", "N/A"),
                "Tool/Function Calling": m.get("tool_calling", "N/A")
            })
        return results

    except Exception as e:
        return [{
            "Model Name": "Discovery Error",
            "Description": f"Failed to perform research discovery: {str(e)}\n\nOutput was: {raw_content if 'raw_content' in locals() else 'None'}",
            "Parameters": "!",
            "Key Features": "!",
            "Tool/Function Calling": "!"
        }]
