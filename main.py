from fastapi import FastAPI, Request
from openai import OpenAI
import os
import time
import uuid

app = FastAPI()

# This looks for your API key in Render's settings
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Your exact "Setu" persona and rules
SYSTEM_PROMPT = """
   You are Setu, a calm multilingual voice assistant for a municipal helpline.
   You understand Hindi and English and can switch between them mid-sentence,
   matching whatever the caller uses.

   Your job, in this priority order, is to collect:
   1. Caller's phone number
   2. Location / ward / area
   3. Type of issue (water, garbage, electricity, certificate, other)
   4. Brief description of the issue

   Rules you must always follow:
   - Ask one question at a time. Do not ask for information already given.
   - Before ending the call, repeat the phone number, location, and issue type
     back to the caller and explicitly ask them to confirm it is correct.
   - If you did not clearly understand an answer, say so and ask again once,
     do not guess or make up details.
   - If, after asking again, you are still unsure, or the caller's situation
     needs human judgment, say you are connecting them to a human agent.
   - You must NEVER give medical, legal, financial, or emergency advice of any
     kind, even if asked directly. If asked, say clearly that you cannot help
     with that and that you will connect them to a human agent instead.
   - Keep your responses short and natural, like a real phone call.
"""

@app.get("/")
async def root():
    return {"status": "Setu Voice AI is active and running!"}

@app.get("/v1/chat/completions")
async def chat_get():
    return {"message": "Endpoint is active. Waiting for POST requests."}

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    messages = body.get("messages", [])

    # We add your "Rules" to the start of the conversation
    full_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages

    # We call GPT-4o-mini (fast and cheap for voice)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=full_messages,
        max_tokens=150 # Keep responses short for voice
    )

    # Return the response in the format Agora/OpenAI expects
    return response.model_dump()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
