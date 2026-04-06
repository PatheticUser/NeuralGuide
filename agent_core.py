import json
from openai import OpenAI
from pydantic import BaseModel, Field
from config import OPENAI_API_KEY

class ModelInfo(BaseModel):
    model_name: str = Field(description="Official name and version")
    description: str = Field(description="1-2 sentence summary of strengths for agentic use")
    parameters: str = Field(description="Size in billions (e.g., 8B, 70B)")
    key_features: str = Field(description="Reasoning capabilities, context window, and multimodal support")
    tool_calling: str = Field(description="Native support for parallel tool execution")

class ModelsResponse(BaseModel):
    models: list[ModelInfo]

def get_agentic_models_from_cloud(query: str) -> list[dict]:
    """
    Calls OpenAI's Chat Completions API using Structured Outputs (Responses API).
    Uses few-shot prompting to ensure a JSON response matching the structured fields.
    """
    if not OPENAI_API_KEY:
        return [{"Model Name": "Error", "Description": "OPENAI_API_KEY not set in .env", "Parameters": "N/A", "Key Features": "N/A", "Tool/Function Calling": "N/A"}]

    client = OpenAI(api_key=OPENAI_API_KEY)

    system_prompt = (
        "You are a Senior AI Engineer specializing in autonomous agent architectures. "
        "Your task is to evaluate and recommend Large Language Models (LLMs) suitable for specific agent use cases. "
        "Feel free to use available up-to-date knowledge (simulated web search logic) to provide accurate details. "
        "Return a JSON format matching the requested schema exactly. "
        "Only recommend models that are highly suitable for the user's specific use case.\n\n"
        "Few-shot Example:\n"
        "User: Recommend a model for coding agents.\n"
        "Output: { \"models\": [ { \"model_name\": \"gpt-4o\", \"description\": \"High-performance multimodal model excellent for coding and reasoning tasks.\", \"parameters\": \"Unknown\", \"key_features\": \"128k context, strong logic, vision capabilities\", \"tool_calling\": \"Excellent parallel tool calling support\" } ] }"
    )

    try:
        completion = client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Recommend models for this use case: {query}"}
            ],
            response_format=ModelsResponse,
        )
        
        models_data = completion.choices[0].message.parsed
        
        # Convert Pydantic model back to a format suitable for the UI
        result = []
        for m in models_data.models:
            result.append({
                "Model Name": m.model_name,
                "Description": m.description,
                "Parameters": m.parameters,
                "Key Features": m.key_features,
                "Tool/Function Calling": m.tool_calling
            })
        return result
        
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return [{
            "Model Name": "OpenAI Error",
            "Description": str(e),
            "Parameters": "N/A",
            "Key Features": "N/A",
            "Tool/Function Calling": "N/A"
        }]
