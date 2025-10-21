"""Generate a response using an LLM."""

from openai import OpenAI

from ai_exercise.constants import SETTINGS, openai_client

def get_completion(client: OpenAI, messages: list[dict], model: str) -> str:
    """Get completion from OpenAI using chat messages format"""
    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    return response.choices[0].message.content
