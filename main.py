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
    title="Setu Municipal Helpline Backend Tools",
    description="Backend tool execution server providing create_ticket and transfer_to_human for Setu Voice AI",
    version="2.2.0"
)

logger = logging.getLogger("uvicorn.error")

# Allowed issue types for municipal helpline
ALLOWED_ISSUE_TYPES = {"water", "garbage", "electricity", "certificate", "other"}

WORD_TO_DIGIT = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9"
}

# ------------------------------------------------------------------------------
# Pydantic Request Models for Backend Tools
# ------------------------------------------------------------------------------
class CreateTicketRequest(BaseModel):
    phone: str = Field(..., description="10-digit Indian mobile number of caller")
    location: str = Field(..., description="Location, ward, or area name")
    issue_type: str = Field(..., description="Type of issue: water, garbage, electricity, certificate, other")
    description: str = Field(..., description="Brief description of the issue")


class TransferToHumanRequest(BaseModel):
    reason: str = Field(..., description="Reason for escalating to a human agent")
    summary: str = Field(..., description="Brief summary of conversation state and caller situation")


# ------------------------------------------------------------------------------
# Validation Helpers
# ------------------------------------------------------------------------------
def validate_indian_phone(text: str) -> tuple[bool, str]:
    """
    Validates an Indian 10-digit mobile number starting with 6, 7, 8, or 9.
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


def validate_issue_type(issue_type: str) -> tuple[bool, str]:
    """Validates issue_type against allowed demo values."""
    if not issue_type or not isinstance(issue_type, str):
        return False, ""

    normalized = issue_type.strip().lower()
    if normalized in ALLOWED_ISSUE_TYPES:
        return True, normalized
    return False, normalized


# ------------------------------------------------------------------------------
# Tool 1: create_ticket Implementation
# ------------------------------------------------------------------------------
def execute_create_ticket(data: CreateTicketRequest) -> Dict[str, Any]:
    """
    Validates create_ticket parameters and creates a real ticket in Freshdesk.
    """
    # 1. Validate Phone Number
    phone_valid, clean_phone = validate_indian_phone(data.phone)
    if not phone_valid:
        logger.warning(f"create_ticket validation failed: Invalid phone number '{data.phone}'")
        return {
            "success": False,
            "error": "invalid_phone_number",
            "message": f"'{data.phone}' is not a valid 10-digit Indian mobile number starting with 6, 7, 8, or 9."
        }

    # 2. Validate Location
    clean_location = data.location.strip() if data.location else ""
    if not clean_location:
        return {
            "success": False,
            "error": "invalid_location",
            "message": "Location parameter cannot be empty."
        }

    # 3. Validate Issue Type
    issue_valid, clean_issue_type = validate_issue_type(data.issue_type)
    if not issue_valid:
        return {
            "success": False,
            "error": "invalid_issue_type",
            "message": f"'{data.issue_type}' is not a recognized issue type. Allowed: water, garbage, electricity, certificate, other."
        }

    # 4. Validate Description
    clean_description = data.description.strip() if data.description else ""
    if not clean_description:
        return {
            "success": False,
            "error": "invalid_description",
            "message": "Description parameter cannot be empty."
        }

    # 5. Read Freshdesk Credentials from Environment Variables
    raw_domain = os.environ.get("FRESHDESK_DOMAIN", "").strip()
    freshdesk_key = os.environ.get("FRESHDESK_API_KEY", "").strip()

    if not raw_domain or not freshdesk_key:
        logger.error("Freshdesk API configuration missing (FRESHDESK_DOMAIN or FRESHDESK_API_KEY not set).")
        return {
            "success": False,
            "error": "freshdesk_config_missing",
            "message": "Freshdesk credentials (FRESHDESK_DOMAIN / FRESHDESK_API_KEY) are not set in environment variables."
        }

    freshdesk_domain = raw_domain.replace("https://", "").replace("http://", "").rstrip("/")
    if not freshdesk_domain.endswith(".freshdesk.com"):
        freshdesk_domain = f"{freshdesk_domain}.freshdesk.com"

    freshdesk_url = f"https://{freshdesk_domain}/api/v2/tickets"

    payload = {
        "subject": clean_issue_type,
        "description": f"Phone: {clean_phone}\nLocation: {clean_location}\nIssue Type: {clean_issue_type}\nDescription: {clean_description}",
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
                "message": f"Complaint registered successfully. Ticket ID is {ticket_id}."
            }
        else:
            logger.error(f"Freshdesk API error status {response.status_code}: {response.text}")
            return {
                "success": False,
                "error": "ticket_creation_failed",
                "message": f"Freshdesk ticket creation failed with status code {response.status_code}."
            }

    except Exception as e:
        logger.exception("Failed to connect to Freshdesk API")
        return {
            "success": False,
            "error": "ticket_creation_failed",
            "message": f"Failed to connect to Freshdesk server: {str(e)}"
        }


# ------------------------------------------------------------------------------
# Tool 2: transfer_to_human Implementation
# ------------------------------------------------------------------------------
def execute_transfer_to_human(data: TransferToHumanRequest) -> Dict[str, Any]:
    """
    Validates parameters, logs the human escalation clearly to the console, and returns success.
    """
    clean_reason = data.reason.strip() if data.reason else ""
    clean_summary = data.summary.strip() if data.summary else ""

    if not clean_reason:
        return {
            "success": False,
            "error": "invalid_reason",
            "message": "Reason parameter cannot be empty."
        }

    if not clean_summary:
        return {
            "success": False,
            "error": "invalid_summary",
            "message": "Summary parameter cannot be empty."
        }

    # Console Logging
    print("\n========== HUMAN ESCALATION ==========")
    print(f"Reason:  {clean_reason}")
    print(f"Summary: {clean_summary}")
    print("=======================================\n")

    logger.info(f"HUMAN ESCALATION | Reason: {clean_reason} | Summary: {clean_summary}")

    return {
        "success": True,
        "status": "human_escalation_requested",
        "message": "Human escalation request logged successfully."
    }


# ------------------------------------------------------------------------------
# FastAPI Endpoints
# ------------------------------------------------------------------------------
@app.get("/")
async def root():
    return {
        "status": "Setu Supporting Tools Backend is active!",
        "architecture": "Agora Conversational AI Backend Tool Execution Server",
        "tools": ["create_ticket", "transfer_to_human"]
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


# Tool 1 Endpoint: create_ticket
@app.post("/tools/create_ticket")
@app.post("/api/tools/create_ticket")
@app.post("/create_ticket")
async def create_ticket_endpoint(request: Request):
    """
    Tool endpoint called when Agora Conversational AI invokes create_ticket.
    """
    body = await request.json()

    # Unwrap arguments if wrapped in OpenAI/MCP function call format
    if "arguments" in body:
        args = body["arguments"]
        if isinstance(args, str):
            args = json.loads(args)
        body = args
    elif "parameters" in body:
        body = body["parameters"]

    try:
        ticket_req = CreateTicketRequest(**body)
    except Exception as e:
        return {
            "success": False,
            "error": "invalid_request_format",
            "message": f"Missing or invalid parameters for create_ticket: {str(e)}"
        }

    return execute_create_ticket(ticket_req)


# Tool 2 Endpoint: transfer_to_human
@app.post("/tools/transfer_to_human")
@app.post("/api/tools/transfer_to_human")
@app.post("/transfer_to_human")
async def transfer_to_human_endpoint(request: Request):
    """
    Tool endpoint called when Agora Conversational AI invokes transfer_to_human.
    """
    body = await request.json()

    # Unwrap arguments if wrapped in OpenAI/MCP function call format
    if "arguments" in body:
        args = body["arguments"]
        if isinstance(args, str):
            args = json.loads(args)
        body = args
    elif "parameters" in body:
        body = body["parameters"]

    try:
        transfer_req = TransferToHumanRequest(**body)
    except Exception as e:
        return {
            "success": False,
            "error": "invalid_request_format",
            "message": f"Missing or invalid parameters for transfer_to_human: {str(e)}"
        }

    return execute_transfer_to_human(transfer_req)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
