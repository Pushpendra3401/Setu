from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import os
import re
import requests
import logging

app = FastAPI(
    title="Setu Municipal Helpline Backend",
    description="Business logic, data validation, and Freshdesk ticket creation tools for Setu Voice AI",
    version="2.0.0"
)

logger = logging.getLogger("uvicorn.error")

# Allowed issue types for municipal helpline demo
ALLOWED_ISSUE_TYPES = {"water", "garbage", "electricity", "certificate", "other"}

# 1. Clean Internal Conversation / Ticket Request Model
class TicketCreateRequest(BaseModel):
    phone: str = Field(..., description="10-digit Indian mobile number of caller")
    location: str = Field(..., description="Location, ward, or area name")
    issue_type: str = Field(..., description="Type of issue: water, garbage, electricity, certificate, other")
    description: str = Field(..., description="Brief description of the issue")
    confirmation_status: Optional[str] = Field("confirmed", description="Confirmation status from caller")


WORD_TO_DIGIT = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9"
}

def validate_indian_phone(phone: str) -> tuple[bool, str]:
    """
    Validates an Indian 10-digit mobile number.
    Accepts numeric strings and spoken word formats (e.g., 'seven eight78331909').
    Returns (is_valid, cleaned_10_digit_phone)
    """
    if not phone or not isinstance(phone, str):
        return False, ""

    text = phone.lower().strip()

    # Replace word numbers with digits
    for word, digit in WORD_TO_DIGIT.items():
        text = re.sub(r'\b' + word + r'\b', digit, text)

    # Extract all digits
    digits = "".join(re.findall(r"\d", text))

    # Strip leading country code prefixes
    if digits.startswith("91") and len(digits) == 12:
        digits = digits[2:]
    elif digits.startswith("0") and len(digits) == 11:
        digits = digits[1:]

    # Verify exactly 10 digits starting with digits 6, 7, 8, or 9
    if re.match(r"^[6-9]\d{9}$", digits):
        return True, digits

    return False, ""


def validate_issue_type(issue_type: str) -> tuple[bool, str]:
    """Validates issue_type against allowed demo values."""
    if not issue_type or not isinstance(issue_type, str):
        return False, ""

    normalized = issue_type.strip().lower()
    if normalized in ALLOWED_ISSUE_TYPES:
        return True, normalized
    return False, normalized


def execute_create_ticket(data: TicketCreateRequest) -> Dict[str, Any]:
    """
    Validates parameters and creates a real ticket in Freshdesk.
    """
    # 1. Validate Phone Number
    phone_valid, clean_phone = validate_indian_phone(data.phone)
    if not phone_valid:
        return {
            "success": False,
            "error": "INVALID_PHONE_NUMBER",
            "message": f"'{data.phone}' is not a valid 10-digit Indian mobile number. Must be a 10-digit number starting with 6, 7, 8, or 9."
        }

    # 2. Validate Issue Type
    issue_valid, clean_issue_type = validate_issue_type(data.issue_type)
    if not issue_valid:
        return {
            "success": False,
            "error": "INVALID_ISSUE_TYPE",
            "message": f"'{data.issue_type}' is not a recognized issue type. Allowed values are: water, garbage, electricity, certificate, other."
        }

    # 3. Read Freshdesk Credentials from Environment Variables
    raw_domain = os.environ.get("FRESHDESK_DOMAIN", "").strip()
    freshdesk_key = os.environ.get("FRESHDESK_API_KEY", "").strip()

    if not raw_domain or not freshdesk_key:
        logger.error("Freshdesk credentials missing in environment variables.")
        return {
            "success": False,
            "error": "FRESHDESK_CONFIG_MISSING",
            "message": "Freshdesk credentials (FRESHDESK_DOMAIN or FRESHDESK_API_KEY) are not configured on the Setu backend server."
        }

    # Clean domain URL
    freshdesk_domain = raw_domain.replace("https://", "").replace("http://", "").rstrip("/")
    if not freshdesk_domain.endswith(".freshdesk.com"):
        freshdesk_domain = f"{freshdesk_domain}.freshdesk.com"

    freshdesk_url = f"https://{freshdesk_domain}/api/v2/tickets"

    payload = {
        "subject": f"Municipal Helpline [{clean_issue_type.upper()}]: Reported in {data.location.strip()}",
        "description": f"Phone: {clean_phone}\nLocation: {data.location.strip()}\nIssue Type: {clean_issue_type}\nDescription: {data.description.strip()}\nStatus: Confirmed by caller",
        "email": f"caller_{clean_phone}@setu-helpline.local",
        "phone": clean_phone,
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
            logger.info(f"Freshdesk ticket #{ticket_id} created successfully for phone {clean_phone}")
            return {
                "success": True,
                "ticket_id": ticket_id,
                "message": f"Complaint successfully registered in Freshdesk with Ticket ID #{ticket_id}.",
                "data": {
                    "ticket_id": ticket_id,
                    "phone": clean_phone,
                    "location": data.location.strip(),
                    "issue_type": clean_issue_type,
                    "description": data.description.strip()
                }
            }
        else:
            logger.error(f"Freshdesk API error {response.status_code}: {response.text}")
            return {
                "success": False,
                "error": "FRESHDESK_API_ERROR",
                "message": f"Freshdesk returned status code {response.status_code}: {response.text}"
            }

    except Exception as e:
        logger.exception("Failed to connect to Freshdesk API")
        return {
            "success": False,
            "error": "FRESHDESK_CONNECTION_FAILED",
            "message": f"Connection to Freshdesk failed: {str(e)}"
        }


@app.get("/")
async def root():
    return {
        "status": "Setu Business Logic & Freshdesk Integration Backend is active!",
        "architecture": "Agora Conversational AI Tool Execution Backend",
        "endpoints": ["/tools/create_ticket", "/health"]
    }

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/v1/chat/completions")
async def chat_get():
    return {"message": "Setu Supporting Backend is active. Tools available at /tools/create_ticket"}

@app.post("/v1/chat/completions")
async def chat_completions_fallback(request: Request):
    """
    Fallback handler if Agora or a client calls /v1/chat/completions.
    Checks if it is a tool execution request or returns a status message.
    """
    body = await request.json()

    # If a tool call arrived via function call format
    if "phone" in body or "arguments" in body or "parameters" in body:
        return await create_ticket_endpoint(request)

    return {
        "status": "Setu Business Logic Server",
        "message": "Agora Conversational AI is running in Managed Mode. Tools are available at /tools/create_ticket."
    }

# 2. Tool / Function Execution Endpoint
@app.post("/tools/create_ticket")
@app.post("/api/tools/create_ticket")
@app.post("/create_ticket")
async def create_ticket_endpoint(request: Request):
    """
    Tool/function endpoint called when Agora Conversational AI invokes create_ticket.
    Supports direct JSON bodies and function call wrapper payloads.
    """
    body = await request.json()

    # Handle wrapper formats if parameters arrive wrapped inside 'arguments' or 'parameters'
    if "arguments" in body:
        args = body["arguments"]
        if isinstance(args, str):
            import json
            args = json.loads(args)
        body = args
    elif "parameters" in body:
        body = body["parameters"]

    try:
        ticket_req = TicketCreateRequest(**body)
    except Exception as e:
        return {
            "success": False,
            "error": "INVALID_REQUEST_FORMAT",
            "message": f"Missing or invalid parameters: {str(e)}"
        }

    return execute_create_ticket(ticket_req)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
