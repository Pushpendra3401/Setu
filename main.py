from fastapi import FastAPI, Request
import time
import uuid

app = FastAPI()

@app.get("/")
async def root():
    return {"status": "Setu Voice AI is active and running!", "endpoint": "/v1/chat/completions"}

@app.get("/v1/chat/completions")
async def chat_get():
    return {"message": "This endpoint is for Agora to send VOICE data via POST. It is working correctly!"}

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    messages = body.get("messages", [])

    # We check how many times the user has spoken to decide which hardcoded reply to send.
    # This simulates "Real Logic" for your end-to-end test.
    user_messages = [m for m in messages if m.get("role") == "user"]

    if len(user_messages) == 0:
        # The very first thing the AI says (if greeting_message isn't used)
        reply = "Namaste! I am Setu, your municipal helpline assistant. Could you please tell me your phone number to start?"
    elif len(user_messages) == 1:
        # After the user gives their phone number
        reply = "Thank you. Which ward or area are you calling from?"
    else:
        # A generic reply for any further conversation during this test
        reply = "I've noted that down. What is the type of issue you are facing? (Water, garbage, or electricity?)"

    return {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "setu-voice-ai",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": reply
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
