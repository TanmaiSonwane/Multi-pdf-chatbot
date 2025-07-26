"""from huggingface_hub import InferenceClient

client = InferenceClient("Qwen/Qwen3-0.6B", token="your_token")
output = client.text_generation("What is LangChain?", max_new_tokens=100)
print(output)"""


import google.generativeai as genai

genai.configure(api_key="AIzaSyAOUIV9e-hP4yrRPT5N9UgL0fI_f3LuaOs")  # Paste your actual key here

models = genai.list_models()
for model in models:
    print(f"Model Name: {model.name}, Supported methods: {model.supported_generation_methods}")

