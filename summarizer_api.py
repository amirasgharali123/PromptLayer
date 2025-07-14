from fastapi import FastAPI, Request
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from promptlayer import PromptLayer
from langsmith.run_helpers import traceable

load_dotenv()

os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "Summarization-Tracer")
os.environ["LANGSMITH_TRACING_V2"] = "true"

promptlayer_client = PromptLayer()
OpenAI = promptlayer_client.openai.OpenAI
client = OpenAI()

app = FastAPI()

class TextInput(BaseModel):
    text: str

@app.post("/summarize/")
@traceable(name="summarization_endpoint")
async def summarize(input_data: TextInput, request: Request):
    print("input_data.text: ",input_data.text)
    completion = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that summarizes text."},
            {"role": "user", "content": f"Summarize this: {input_data.text}"}
        ],
        pl_tags=["summarizer", "v2"]
    )
    summary = completion.choices[0].message.content
    return {"summary": summary}
