from openai import OpenAI
import os

try:    
    client = OpenAI(
            api_key = os.environ["OPENAI_API_KEY"]
        )
except:
    raise EnvironmentError("No OpenAI API key found.")