"""Configuration for the LLM Council."""

import os
from dotenv import load_dotenv

load_dotenv()

# OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")

# Council members - list of OpenAI model identifiers
COUNCIL_MODELS = [
    "gpt-4.1",
    "gpt-4o",
    "gpt-4.1-mini",
    "gpt-4o-mini",
]

# Chairman model - synthesizes final response
CHAIRMAN_MODEL = "gpt-4.1"

# OpenAI Responses API endpoint
OPENAI_API_URL = "https://api.openai.com/v1/responses"

# Data directory for conversation storage
DATA_DIR = "data/conversations"
