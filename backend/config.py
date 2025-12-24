"""Configuration for the LLM Council."""

import os
from dotenv import load_dotenv

load_dotenv()

# OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")

# Council members - list of OpenAI model identifiers
# You can override the default list by setting the `COUNCIL_MODELS` env var
# as a comma-separated list, e.g. COUNCIL_MODELS="gpt-4.1,gpt-5-mini"
env_models = os.getenv("COUNCIL_MODELS")
if env_models:
    COUNCIL_MODELS = [m.strip() for m in env_models.split(",") if m.strip()]
else:
    COUNCIL_MODELS = [
        "gpt-4.1",
        "gpt-4o",
        "gpt-4.1-mini",
        "gpt-4o-mini",
        # newer / experimental models added to defaults
        "gpt-4o-realtime-preview",
        "gpt-5-mini",
    ]

# Chairman model - synthesizes final response (can be overridden via env)
CHAIRMAN_MODEL = os.getenv("CHAIRMAN_MODEL", "gpt-4.1")

# OpenAI Responses API endpoint (can be overridden in .env)
OPENAI_API_URL = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/responses")

# Data directory for conversation storage (can be overridden in .env)
DATA_DIR = os.getenv("DATA_DIR", "data/conversations")
