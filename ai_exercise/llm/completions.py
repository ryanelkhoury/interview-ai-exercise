"""Generate a response using an LLM."""

from openai import OpenAI

from ai_exercise.constants import SETTINGS, openai_client


def complete(prompt: str) -> str:
    """Simple completion function that takes a prompt and returns a response."""
    messages = [{"role": "user", "content": prompt}]
    return get_completion(openai_client, messages, SETTINGS.openai_model)


def create_prompt(query: str, context: list[str]) -> str:
    """Create a prompt combining query and context"""
    context_str = "\n\n".join(context)
    return f"""Please answer the question based on the following context:

Context:
{context_str}

Question: {query}

Answer:"""


def get_completion(client: OpenAI, messages: list[dict], model: str) -> str:
    """Get completion from OpenAI using chat messages format"""
    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    return response.choices[0].message.content
