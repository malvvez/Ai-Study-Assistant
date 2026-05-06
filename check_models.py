"""You can check here what versions of google AI models are available for generation. This is a simple test to ensure that the API key is working and that we can connect to the Google GenAI service."""

import google.generativeai as genai

genai.configure(api_key="your_api_key_here")

for m in genai.list_models():
    if "generateContent" in m.supported_generation_methods:
        print(m.name)