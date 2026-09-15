"""
Gemini client used ONLY for the rendering step in generate_synthetic_data.py
(JSON -> messy invoice text). The roundtrip quality check and all production
extraction still run on GroqClient - this keeps training-data generation and
the benchmark baseline on different model families/providers.
"""
import os
from google import genai
from google.genai import types


class GeminiClient:
    def __init__(self, model_name: str = "gemini-3.1-flash-lite"):
        api_key = os.environ["GEMINI_API_KEY"]
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def extract_raw(self, prompt: str) -> str:
        """
        Synchronous call, matching GroqClient's interface used elsewhere
        in this script. Returns plain text - NOT JSON - since this client
        is only used for the rendering step (JSON -> messy invoice text).
        """
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.7,  # variance helps synthetic data diversity
            ),
        )
        return response.text