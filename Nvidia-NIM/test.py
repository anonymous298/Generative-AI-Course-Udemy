import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

os.environ['NVIDIA_NIM_API_KEY'] = os.getenv('NVIDIA_NIM_API_KEY')

client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = os.getenv('NVIDIA_NIM_API_KEY')
)

completion = client.chat.completions.create(
  model="meta/llama-3.3-70b-instruct",
  messages=[{"role":"user","content":"Hi, How are you?"}],
  temperature=0.2,
  top_p=0.7,
  max_tokens=1024,
  stream=True
)

for chunk in completion:
  if chunk.choices[0].delta.content is not None:
    print(chunk.choices[0].delta.content, end="")
