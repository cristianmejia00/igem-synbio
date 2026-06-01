"""Prompt loading, the OpenAI client, and generic chat helpers."""
import yaml
from openai import OpenAI

from .paths import OPENAI_KEY_PATH, OPENAI_MODEL, PROMPTS_PATH


def load_prompts(path=PROMPTS_PATH) -> dict:
    """Load the prompt templates and theme definition from ``prompts.yaml``."""
    with open(path) as f:
        return yaml.safe_load(f)


def make_client(key_path=OPENAI_KEY_PATH) -> OpenAI:
    """Read the API key file and return an authenticated OpenAI client."""
    api_key = open(key_path).read().strip()
    return OpenAI(api_key=api_key)


def ask_gpt(
    client: OpenAI,
    system_prompt: str,
    user_prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.1,
    max_tokens: int = 500,
) -> str:
    """Send a system + user prompt to OpenAI and return the assistant reply."""
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content


def fmt_prompt(template: str, **kwargs) -> str:
    """Format a prompt template by replacing ``{key}`` placeholders."""
    result = template
    for key, value in kwargs.items():
        result = result.replace("{" + key + "}", str(value))
    return result
