from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import json
import time
import uuid
import asyncio

app = FastAPI()

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
    stream = body.get("stream", False)

    response_text = "Namaste! This is Setu, your municipal helpline voice assistant. How can I assist you today?"
    completion_id = f"chatcmpl-{uuid.uuid4()}"
    created_time = int(time.time())

    # Agora Conversational AI Engine requests stream: True for real-time TTS audio generation
    if stream:
        async def sse_generator():
            # First chunk: Send response text content
            chunk1 = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created_time,
                "model": "setu-voice-ai",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": response_text},
                        "finish_reason": None
                    }
                ]
            }
            yield f"data: {json.dumps(chunk1)}\n\n"
            await asyncio.sleep(0.05)

            # Final chunk: Indicate completion stop
            chunk2 = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created_time,
                "model": "setu-voice-ai",
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop"
                    }
                ]
            }
            yield f"data: {json.dumps(chunk2)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(sse_generator(), media_type="text/event-stream")

    # Non-streaming JSON response fallback
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created_time,
        "model": "setu-voice-ai",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text
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
