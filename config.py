"""
Shared configuration and clients for the application.
"""

import os
from openai import OpenAI
from openai import AsyncOpenAI
from google import genai
from dotenv import load_dotenv


# Load environment variables
load_dotenv()

# Initialize OpenAI client
OPEN_AI_API_KEY = os.getenv("OPEN_AI_API_KEY")
client = OpenAI(api_key=OPEN_AI_API_KEY)
async_client = AsyncOpenAI(api_key=OPEN_AI_API_KEY)

# Initialize Gemini client
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
gemini_client = genai.Client(api_key=GEMINI_API_KEY)
