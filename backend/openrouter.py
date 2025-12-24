"""OpenAI Responses API client for making LLM requests."""

import httpx
from typing import List, Dict, Any, Optional
from .config import OPENAI_API_KEY, OPENAI_API_URL


def _messages_to_responses_input(
    messages: List[Dict[str, str]]
) -> List[Dict[str, Any]]:
    """Convert chat messages to OpenAI Responses API input format."""
    input_items = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content") or ""
        input_items.append({
            "role": role,
            "content": [
                {"type": "input_text", "text": content}
            ],
        })
    return input_items


def _extract_output_text(data: Dict[str, Any]) -> Optional[str]:
    """Extract output text from a Responses API response payload."""
    output_text = data.get("output_text")
    if isinstance(output_text, str) and output_text:
        return output_text

    output = data.get("output", [])
    parts = []
    for item in output:
        for content in item.get("content", []):
            if content.get("type") in ("output_text", "text"):
                text = content.get("text")
                if text:
                    parts.append(text)
    if parts:
        return "".join(parts)

    return None


async def query_model(
    model: str,
    messages: List[Dict[str, str]],
    timeout: float = 120.0
) -> Optional[Dict[str, Any]]:
    """
    Query a single model via OpenAI Responses API.

    Args:
        model: OpenAI model identifier (e.g., "gpt-4o")
        messages: List of message dicts with 'role' and 'content'
        timeout: Request timeout in seconds

    Returns:
        Response dict with 'content' and optional 'reasoning_details', or None if failed
    """
    if not OPENAI_API_KEY:
        print("Error querying model: OPENAI_API_KEY is not set")
        return None

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "input": _messages_to_responses_input(messages),
    }

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                OPENAI_API_URL,
                headers=headers,
                json=payload
            )
            response.raise_for_status()

            data = response.json()
            output_text = _extract_output_text(data)
            if output_text is None:
                raise ValueError("No output_text found in response payload")

            return {
                'content': output_text,
                'reasoning_details': None,
            }

    except Exception as e:
        print(f"Error querying model {model}: {e}")
        return None


async def query_models_parallel(
    models: List[str],
    messages: List[Dict[str, str]]
) -> Dict[str, Optional[Dict[str, Any]]]:
    """
    Query multiple models in parallel.

    Args:
        models: List of OpenAI model identifiers
        messages: List of message dicts to send to each model

    Returns:
        Dict mapping model identifier to response dict (or None if failed)
    """
    import asyncio

    # Create tasks for all models
    tasks = [query_model(model, messages) for model in models]

    # Wait for all to complete
    responses = await asyncio.gather(*tasks)

    # Map models to their responses
    return {model: response for model, response in zip(models, responses)}
