from fastapi import FastAPI, Request
import time
import uuid

app = FastAPI()

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    # Reads the incoming JSON request containing the conversation messages
    body = await request.json()
    messages = body.get("messages", [])

    # Returns a hardcoded reply strictly formatted to match OpenAI's Chat Completion API structure
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
                    "content": "Namaste! This is Setu, your municipal helpline voice assistant. How can I assist you today?"
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 15,
            "total_tokens": 25
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
