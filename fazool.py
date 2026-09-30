import os
from groq import Groq

# Groq client initialize karein
client = Groq(api_key=os.environ.get("GROQ_API_KEY", "your_groq_api_key_here"))

# Models ki list print karein
models = client.models.list()

print("Available Groq Models:")
for model in models.data:
    print(f"- {model.id}")
    
print("end of list")       