from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import os
import re
import requests
import json
import time
import uuid
import logging

app = FastAPI(
    title="Setu Municipal Helpline Backend",
    description="Stateful Conversation Tracking, Business Logic, and Freshdesk Integration Server",
    version="2.1.0"
)

logger = logging.getLogger("uvicorn.error")

# Allowed issue types for municipal helpline
ALLOWED_ISSUE_TYPES = {"water", "garbage", "electricity", "certificate", "other"}

# Word-to-digit map for Indian phone number parsing
WORD_TO_DIGIT = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9"
}

# ------------------------------------------------------------------------------
# In-Memory Conversation Database (Keyed by conversation_id)
# ------------------------------------------------------------------------------
conversations_db: Dict[str, Dict[str, Any]] = {}


def get_or_create_conversation(conversation_id: str) -> Dict[str, Any]:
    """Retrieves or initializes conversation state dictionary."""
    if conversation_id not in conversations_db:
        conversations_db[conversation_id] = {
            "conversation_id": conversation_id,
            "phone": None,
            "phone_confidence": "low",
            "location": None,
            "location_confidence": "low",
            "issue_type": None,
            "issue_type_confidence": "low",
            "description": None,
            "description_confidence": "low",
            "confirmed": False,
            "ticket_id": None,
            "updated_at": time.time()
        }
    return conversations_db[conversation_id]


# ------------------------------------------------------------------------------
# Pydantic Request Models
# ------------------------------------------------------------------------------
class TicketCreateRequest(BaseModel):
    phone: str = Field(..., description="10-digit Indian mobile number of caller")
    location: str = Field(..., description="Location, ward, or area name")
    issue_type: str = Field(..., description="Type of issue: water, garbage, electricity, certificate, other")
    description: str = Field(..., description="Brief description of the issue")
    confirmation_status: Optional[str] = Field("confirmed", description="Confirmation status from caller")


class ProcessMessageRequest(BaseModel):
    conversation_id: str
    message: str


# ------------------------------------------------------------------------------
# Helper Functions: Validation & Field Extraction
# ------------------------------------------------------------------------------
def validate_indian_phone(text: str) -> tuple[bool, str]:
    """
    Validates an Indian 10-digit mobile number from string/spoken words.
    Returns (is_valid, cleaned_10_digit_phone)
    """
    if not text or not isinstance(text, str):
        return False, ""

    lowered = text.lower().strip()

    # Convert spoken word digits to numbers
    for word, digit in WORD_TO_DIGIT.items():
        lowered = re.sub(r'\b' + word + r'\b', digit, lowered)

    # Extract all numeric digits
    digits = "".join(re.findall(r"\d", lowered))

    # Strip leading country code prefixes
    if digits.startswith("91") and len(digits) == 12:
        digits = digits[2:]
    elif digits.startswith("0") and len(digits) == 11:
        digits = digits[1:]

    # Must be exactly 10 digits starting with 6, 7, 8, or 9
    if re.match(r"^[6-9]\d{9}$", digits):
        return True, digits

    return False, digits


def extract_fields_from_text(user_text: str, state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extracts phone, location, issue_type, and description from the latest user message
    and updates conversation state with confidence levels.
    """
    text = user_text.strip()
    lowered = text.lower()

    # 1. Extract & Validate Phone Number
    is_valid_phone, clean_phone = validate_indian_phone(text)
    if is_valid_phone:
        state["phone"] = clean_phone
        state["phone_confidence"] = "high"
    elif len(clean_phone) > 0 and state["phone_confidence"] != "high":
        state["phone_confidence"] = "low"

    # 2. Extract Location
    location_match = re.search(r"\b(ward\s*\d+|jaipur|delhi|mumbai|bangalore|pune|sector\s*\d+|block\s*[a-z0-9]+)\b", lowered)
    if location_match:
        state["location"] = location_match.group(0).title()
        state["location_confidence"] = "high"
    elif state["phone_confidence"] == "high" and state["location_confidence"] != "high" and not is_valid_phone:
        if len(text) >= 3 and not re.search(r"^\d+$", "".join(re.findall(r"\d", text))) and not any(k in lowered for k in ["water", "electricity", "garbage", "certificate", "yes", "no", "hello"]):
            state["location"] = text.title()
            state["location_confidence"] = "high"

    # 3. Extract Issue Type
    if "water" in lowered or "pipe" in lowered or "leak" in lowered or "drain" in lowered:
        state["issue_type"] = "water"
        state["issue_type_confidence"] = "high"
    elif "garbage" in lowered or "trash" in lowered or "waste" in lowered or "clean" in lowered:
        state["issue_type"] = "garbage"
        state["issue_type_confidence"] = "high"
    elif "electricity" in lowered or "power" in lowered or "light" in lowered or "current" in lowered or "generator" in lowered:
        state["issue_type"] = "electricity"
        state["issue_type_confidence"] = "high"
    elif "certificate" in lowered or "birth" in lowered or "death" in lowered or "license" in lowered:
        state["issue_type"] = "certificate"
        state["issue_type_confidence"] = "high"

    # 4. Extract Description
    if state["phone_confidence"] == "high" and state["location_confidence"] == "high" and state["issue_type_confidence"] == "high":
        if lowered not in ["yes", "no", "hello", "ok", "correct", "confirm", "haan", "ha", "yes, correct"]:
            if not is_valid_phone and not location_match:
                state["description"] = text
                state["description_confidence"] = "high"

    # 5. Handle Confirmation
    if lowered in ["yes", "correct", "true", "confirm", "haan", "haa", "ha", "yes correct", "yes, correct"]:
        if (state["phone_confidence"] == "high" and
            state["location_confidence"] == "high" and
            state["issue_type_confidence"] == "high" and
            state["description_confidence"] == "high"):
            state["confirmed"] = True

    state["updated_at"] = time.time()
    return state


def generate_next_response(state: Dict[str, Any]) -> str:
    """
    Generates conversational reply based on missing fields in priority order:
    1. Phone -> 2. Location -> 3. Issue Type -> 4. Description -> 5. Confirmation
    """
    # Priority 1: Phone
    if state["phone_confidence"] != "high":
        return "Namaste! Welcome to Setu municipal helpline. Could you please provide your 10-digit mobile number?"

    # Priority 2: Location
    if state["location_confidence"] != "high":
        return f"Thank you. I have noted your phone number as {state['phone']}. Which ward, area, or location are you calling from?"

    # Priority 3: Issue Type
    if state["issue_type_confidence"] != "high":
        return f"Got it. Location is {state['location']}. What type of issue are you facing? (water, garbage, electricity, certificate, or other)"

    # Priority 4: Description
    if state["description_confidence"] != "high":
        return f"Understood, issue type is {state['issue_type']}. Could you please give a brief description of the problem?"

    # Priority 5: Confirmation
    if not state["confirmed"]:
        return (f"Let me confirm your details: Phone {state['phone']}, Location {state['location']}, "
                f"Issue Type {state['issue_type']}, Description '{state['description']}'. Is this information correct?")

    # Confirmed -> Execute Ticket Creation
    if state["confirmed"] and not state["ticket_id"]:
        ticket_req = TicketCreateRequest(
            phone=state["phone"],
            location=state["location"],
            issue_type=state["issue_type"],
            description=state["description"],
            confirmation_status="confirmed"
        )
        res = execute_create_ticket(ticket_req)
        if res.get("success"):
            state["ticket_id"] = res.get("ticket_id")
            return f"Thank you! Your complaint has been registered in Freshdesk with Ticket ID #{res.get('ticket_id')}."
        else:
            return f"Your details are confirmed, but Freshdesk ticket creation returned: {res.get('message')}"

    return f"Your complaint is already registered under Ticket ID #{state['ticket_id']}. Thank you for calling Setu!"


# ------------------------------------------------------------------------------
# Freshdesk Integration
# ------------------------------------------------------------------------------
def execute_create_ticket(data: TicketCreateRequest) -> Dict[str, Any]:
    """Validates parameters and creates a real ticket in Freshdesk."""
    raw_domain = os.environ.get("FRESHDESK_DOMAIN", "").strip()
    freshdesk_key = os.environ.get("FRESHDESK_API_KEY", "").strip()

    if not raw_domain or not freshdesk_key:
        logger.error("Freshdesk credentials missing in environment variables.")
        return {
            "success": False,
            "error": "FRESHDESK_CONFIG_MISSING",
            "message": "Freshdesk credentials (FRESHDESK_DOMAIN / FRESHDESK_API_KEY) are not set in environment variables."
        }

    freshdesk_domain = raw_domain.replace("https://", "").replace("http://", "").rstrip("/")
    if not freshdesk_domain.endswith(".freshdesk.com"):
        freshdesk_domain = f"{freshdesk_domain}.freshdesk.com"

    freshdesk_url = f"https://{freshdesk_domain}/api/v2/tickets"

    payload = {
        "subject": f"Municipal Helpline [{data.issue_type.upper()}]: Reported in {data.location}",
        "description": f"Phone: {data.phone}\nLocation: {data.location}\nIssue Type: {data.issue_type}\nDescription: {data.description}\nStatus: Confirmed by caller",
        "email": f"caller_{data.phone}@setu-helpline.local",
        "phone": data.phone,
        "priority": 1,
        "status": 2
    }

    try:
        response = requests.post(
            freshdesk_url,
            json=payload,
            auth=(freshdesk_key, "X"),
            headers={"Content-Type": "application/json"},
            timeout=10
        )

        if response.status_code == 201:
            ticket_data = response.json()
            ticket_id = ticket_data.get("id")
            return {
                "success": True,
                "ticket_id": ticket_id,
                "message": f"Complaint successfully registered in Freshdesk with Ticket ID #{ticket_id}.",
                "data": ticket_data
            }
        else:
            return {
                "success": False,
                "error": "FRESHDESK_API_ERROR",
                "message": f"Freshdesk returned status {response.status_code}: {response.text}"
            }
    except Exception as e:
        return {
            "success": False,
            "error": "FRESHDESK_CONNECTION_FAILED",
            "message": f"Connection failed: {str(e)}"
        }


# ------------------------------------------------------------------------------
# FastAPI Endpoints
# ------------------------------------------------------------------------------
@app.get("/")
async def root():
    return {
        "status": "Setu Stateful Conversation Backend is active!",
        "architecture": "In-Memory State Machine & Freshdesk Integration",
        "active_conversations": len(conversations_db)
    }

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/conversation/{conversation_id}")
async def get_conversation_state(conversation_id: str):
    """Retrieves current tracked state for a conversation."""
    if conversation_id not in conversations_db:
        raise HTTPException(status_code=404, detail=f"Conversation ID '{conversation_id}' not found.")
    return conversations_db[conversation_id]


@app.post("/conversation/reset/{conversation_id}")
async def reset_conversation_state(conversation_id: str):
    """Resets state for a conversation ID."""
    if conversation_id in conversations_db:
        del conversations_db[conversation_id]
    return {"status": "reset", "conversation_id": conversation_id}


@app.post("/conversation/process")
async def process_conversation_message(req: ProcessMessageRequest):
    """
    Direct message processing endpoint:
    Takes {conversation_id, message}, extracts fields, updates state, and returns response.
    """
    state = get_or_create_conversation(req.conversation_id)
    extract_fields_from_text(req.message, state)
    reply = generate_next_response(state)
    return {
        "conversation_id": req.conversation_id,
        "reply": reply,
        "state": state
    }


@app.get("/v1/chat/completions")
async def chat_get():
    return {"message": "Setu Supporting Backend is active. Waiting for POST requests."}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """
    OpenAI-compatible completions endpoint:
    Parses messages, extracts fields, updates conversation state, and returns formatted completion.
    """
    body = await request.json()
    messages: List[Dict[str, Any]] = body.get("messages", [])
    conversation_id = body.get("conversation_id", body.get("channel_name", "default_demo"))
    stream = body.get("stream", True)

    state = get_or_create_conversation(conversation_id)

    # Extract user's latest text message
    user_messages = [m for m in messages if m.get("role") == "user"]
    if user_messages:
        latest_text = user_messages[-1].get("content", "")
        extract_fields_from_text(latest_text, state)

    reply_text = generate_next_response(state)
    completion_id = f"chatcmpl-{uuid.uuid4()}"
    created_time = int(time.time())

    if stream:
        async def sse_generator():
            chunk1 = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created_time,
                "model": "setu-voice-ai",
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": reply_text}, "finish_reason": None}]
            }
            yield f"data: {json.dumps(chunk1)}\n\n"
            chunk2 = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created_time,
                "model": "setu-voice-ai",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
            }
            yield f"data: {json.dumps(chunk2)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(sse_generator(), media_type="text/event-stream")

    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created_time,
        "model": "setu-voice-ai",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": reply_text},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
    }


@app.post("/tools/create_ticket")
@app.post("/api/tools/create_ticket")
@app.post("/create_ticket")
async def create_ticket_endpoint(request: Request):
    """Tool execution endpoint called when create_ticket function is invoked."""
    body = await request.json()
    if "arguments" in body:
        args = body["arguments"]
        if isinstance(args, str):
            args = json.loads(args)
        body = args
    elif "parameters" in body:
        body = body["parameters"]

    try:
        ticket_req = TicketCreateRequest(**body)
    except Exception as e:
        return {"success": False, "error": "INVALID_REQUEST_FORMAT", "message": str(e)}

    return execute_create_ticket(ticket_req)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
