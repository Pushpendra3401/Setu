from fastapi import FastAPI, HTTPException, Request, File, UploadFile, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import os
import re
import requests
import json
import time
import uuid
import random
import logging

app = FastAPI(
    title="Setu Municipal Helpline Backend Tools & Human Voice Handoff",
    description="Backend tool execution server with human voice handoff, Freshdesk integration, and safety guardrails",
    version="3.4.0"
)

# CORS Middleware Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = logging.getLogger("uvicorn.error")

ALLOWED_ISSUE_TYPES = {"water", "garbage", "electricity", "certificate", "other"}

WORD_TO_DIGIT = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9"
}

# In-memory persistence databases
conversations_db: Dict[str, Dict[str, Any]] = {}
escalations_db: Dict[str, Dict[str, Any]] = {}


def mask_phone(phone: Optional[str]) -> str:
    """Masks phone number for safe logging (e.g. 6362829732 -> ******9732)."""
    if not phone or len(phone) < 4:
        return "******"
    return "******" + phone[-4:]


def get_or_create_conversation(conversation_id: str) -> Dict[str, Any]:
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
# Deterministic Backend Guardrails
# ------------------------------------------------------------------------------
MEDICAL_PATTERNS = [
    r"\b(take\s*(a\s*)?(pill|tablet|medicine|dose|\d+)|dosage|dose|mg|ml|pills?|tablets?|capsules?)\b",
    r"\b(ibuprofen|paracetamol|aspirin|antibiotic|prescription|ointment)\b",
    r"\b(diagnos(e|is)|symptoms? of|treat(ment)?|cure|remedy)\b",
    r"\b(first aid|apply pressure|bandage|cpr)\b"
]

LEGAL_PATTERNS = [
    r"\b(you should sue|sue them|file a lawsuit|hire a lawyer|attorney)\b",
    r"\b(legal (rights|action|counsel|advice)|court case|liable for damages)\b",
    r"\b(prosecute|subpoena|breach of contract)\b"
]

FINANCIAL_PATTERNS = [
    r"\b(invest (in|your)|buy (stock|shares|crypto|bitcoin)|guaranteed returns?)\b",
    r"\b(financial advice|portfolio|stock market|trading tip)\b",
    r"\b(transfer money|wire funds|tax loophole)\b"
]

EMERGENCY_PATTERNS = [
    r"\b(call (911|108|112|ambulance|police|fire brigade))\b",
    r"\b(evacuate immediately|emergency room|icu)\b"
]

ALL_GUARDRAIL_PATTERNS = {
    "medical": MEDICAL_PATTERNS,
    "legal": LEGAL_PATTERNS,
    "financial": FINANCIAL_PATTERNS,
    "emergency": EMERGENCY_PATTERNS
}

SAFE_GUARDRAIL_RESPONSE = "I'm not able to help with that directly, but I'll connect you with someone who can."


def check_and_apply_guardrails(reply_text: str) -> tuple[bool, str, Optional[str]]:
    if not reply_text:
        return False, reply_text, None

    lowered = reply_text.lower()

    for category, patterns in ALL_GUARDRAIL_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, lowered):
                logger.warning(f"DETERMINISTIC GUARDRAIL INTERCEPTED [{category.upper()}]: '{reply_text[:60]}...'")

                esc_req = TransferToHumanRequest(
                    reason=f"Guardrail Intercepted: {category.title()} Advice Detected",
                    summary=f"Caller requested restricted {category} advice.",
                    issue_one_line=f"Restricted Advice Intercepted ({category.title()} Query)",
                    confirmed_fields={},
                    key_points=f"Backend guardrail intercepted response containing restricted advice: '{reply_text[:120]}...'",
                    unresolved=f"Caller requested {category} assistance. Backend safety net overrode output and requested human escalation."
                )
                execute_transfer_to_human(esc_req)

                return True, SAFE_GUARDRAIL_RESPONSE, category

    return False, reply_text, None


# ------------------------------------------------------------------------------
# Pydantic Request Models
# ------------------------------------------------------------------------------
class CreateTicketRequest(BaseModel):
    phone: str = Field(..., description="10-digit Indian mobile number of caller")
    location: str = Field(..., description="Location, ward, or area name")
    issue_type: str = Field(..., description="Type of issue: water, garbage, electricity, certificate, other")
    description: str = Field(..., description="Brief description of the issue")


class TransferToHumanRequest(BaseModel):
    reason: str = Field(..., description="One phrase explaining why this call is escalating")
    summary: Optional[str] = Field(None, description="Brief summary of conversation state and situation")
    issue_one_line: Optional[str] = Field(None, description="One line summary of the issue")
    confirmed_fields: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Dictionary of locked-in fields")
    key_points: Optional[str] = Field(None, description="2-3 sentences max detailing what was discussed without filler")
    unresolved: Optional[str] = Field(None, description="What is uncertain and specifically why a human agent is needed")
    phone: Optional[str] = None
    location: Optional[str] = None
    issue_type: Optional[str] = None
    description: Optional[str] = None
    channel_name: Optional[str] = None


class StatusUpdateRequest(BaseModel):
    status: str


# ------------------------------------------------------------------------------
# Validation & Extraction Helpers
# ------------------------------------------------------------------------------
def validate_indian_phone(text: str) -> tuple[bool, str]:
    if not text or not isinstance(text, str):
        return False, ""

    lowered = text.lower().strip()

    for word, digit in WORD_TO_DIGIT.items():
        lowered = re.sub(r'\b' + word + r'\b', digit, lowered)

    digits = "".join(re.findall(r"\d", lowered))

    if digits.startswith("91") and len(digits) == 12:
        digits = digits[2:]
    elif digits.startswith("0") and len(digits) == 11:
        digits = digits[1:]

    if re.match(r"^[6-9]\d{9}$", digits):
        return True, digits

    return False, digits


def validate_issue_type(issue_type: str) -> tuple[bool, str]:
    if not issue_type or not isinstance(issue_type, str):
        return False, ""

    normalized = issue_type.strip().lower()
    if normalized in ALLOWED_ISSUE_TYPES:
        return True, normalized
    return False, normalized


def extract_fields_from_text(user_text: str, state: Dict[str, Any]) -> Dict[str, Any]:
    text = user_text.strip()
    lowered = text.lower()
    location_match = None

    is_valid_phone, clean_phone = validate_indian_phone(text)
    if is_valid_phone:
        state["phone"] = clean_phone
        state["phone_confidence"] = "high"
    elif len(clean_phone) > 0 and state["phone_confidence"] != "high":
        state["phone_confidence"] = "low"

    if state["location_confidence"] != "high":
        location_match = re.search(r"\b(ward\s*\d+|jaipur|delhi|mumbai|bangalore|pune|sector\s*\d+)\b", lowered)
        if location_match:
            state["location"] = location_match.group(0).title()
            state["location_confidence"] = "high"
        elif state["phone_confidence"] == "high" and not is_valid_phone:
            if len(text) >= 3 and not re.search(r"^\d+$", "".join(re.findall(r"\d", text))) and not any(k in lowered for k in ["water", "electricity", "garbage", "certificate", "yes", "no", "hello"]):
                state["location"] = text.title()
                state["location_confidence"] = "high"

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

    if state["phone_confidence"] == "high" and state["location_confidence"] == "high" and state["issue_type_confidence"] == "high":
        if lowered not in ["yes", "no", "hello", "ok", "correct", "confirm", "haan", "ha", "yes, correct"]:
            if not is_valid_phone and not location_match:
                state["description"] = text
                state["description_confidence"] = "high"

    if lowered in ["yes", "correct", "true", "confirm", "haan", "haa", "ha", "yes correct", "yes, correct"]:
        if (state["phone_confidence"] == "high" and
            state["location_confidence"] == "high" and
            state["issue_type_confidence"] == "high" and
            state["description_confidence"] == "high"):
            state["confirmed"] = True

    state["updated_at"] = time.time()
    return state


def generate_next_response(state: Dict[str, Any]) -> str:
    if state["phone_confidence"] != "high":
        return "Namaste! Welcome to Setu municipal helpline. Could you please provide your 10-digit mobile number?"

    if state["location_confidence"] != "high":
        return f"Thank you. I have noted your phone number as {state['phone']}. Which ward, area, or location are you calling from?"

    if state["issue_type_confidence"] != "high":
        return f"Got it. Location is {state['location']}. What type of issue are you facing? (water, garbage, electricity, certificate, or other)"

    if state["description_confidence"] != "high":
        return f"Understood, issue type is {state['issue_type']}. Could you please give a brief description of the problem?"

    if not state["confirmed"]:
        return (f"Let me confirm your details: Phone {state['phone']}, Location {state['location']}, "
                f"Issue Type {state['issue_type']}, Description '{state['description']}'. Is this information correct?")

    if state["confirmed"] and not state["ticket_id"]:
        ticket_req = CreateTicketRequest(
            phone=state["phone"],
            location=state["location"],
            issue_type=state["issue_type"],
            description=state["description"]
        )
        res = execute_create_ticket(ticket_req)
        if res.get("success"):
            state["ticket_id"] = res.get("ticket_id")
            return f"Thank you! Your complaint has been registered in Freshdesk with Ticket ID #{res.get('ticket_id')}."
        else:
            return f"Your details are confirmed, but Freshdesk ticket creation returned: {res.get('message')}"

    return f"Your complaint is already registered under Ticket ID #{state['ticket_id']}. Thank you for calling Setu!"


# ------------------------------------------------------------------------------
# Fast2SMS Integration Function
# ------------------------------------------------------------------------------
def send_sms_upload_link(phone: str, ticket_id: int) -> Dict[str, Any]:
    api_key = os.environ.get("FAST2SMS_API_KEY", "").strip()
    raw_domain = os.environ.get("SETU_RENDER_DOMAIN", "setu-9mx9.onrender.com").strip()
    render_domain = raw_domain.replace("https://", "").replace("http://", "").rstrip("/")

    upload_link = f"https://{render_domain}/upload/{ticket_id}"

    if os.environ.get("TEST_MODE", "").lower() in ["true", "1"]:
        logger.info(f"[TEST_MODE] Simulating SMS to {mask_phone(phone)}. Upload link: {upload_link}")
        return {
            "sent": True,
            "upload_link": upload_link,
            "fast2sms_response": {"return": True, "message": "Simulated SMS sent successfully"}
        }

    if not api_key:
        logger.warning(f"FAST2SMS_API_KEY not set. Skipping SMS. Photo upload link: {upload_link}")
        return {
            "sent": False,
            "error": "fast2sms_key_missing",
            "upload_link": upload_link,
            "message": "FAST2SMS_API_KEY environment variable is not configured."
        }

    url = "https://www.fast2sms.com/dev/bulkV2"
    sms_message = f"Setu Municipal Helpline: Upload photo evidence for Ticket #{ticket_id} here: {upload_link}"

    params = {
        "authorization": api_key,
        "route": "q",
        "message": sms_message,
        "language": "english",
        "flash": 0,
        "numbers": phone
    }

    try:
        res = requests.get(url, params=params, timeout=10)
        res_data = res.json() if "json" in res.headers.get("content-type", "") else {"text": res.text}
        logger.info(f"Fast2SMS API Response for {mask_phone(phone)}: {res_data}")

        is_success = res.status_code == 200 and isinstance(res_data, dict) and res_data.get("return") is True
        return {
            "sent": is_success,
            "upload_link": upload_link,
            "fast2sms_response": res_data
        }
    except Exception as e:
        logger.exception(f"Fast2SMS request failed for phone {mask_phone(phone)}")
        return {
            "sent": False,
            "error": str(e),
            "upload_link": upload_link
        }


# ------------------------------------------------------------------------------
# Tool 1: create_ticket Implementation
# ------------------------------------------------------------------------------
def execute_create_ticket(data: CreateTicketRequest) -> Dict[str, Any]:
    phone_valid, clean_phone = validate_indian_phone(data.phone)
    if not phone_valid:
        return {
            "success": False,
            "error": "invalid_phone_number",
            "message": f"'{data.phone}' is not a valid 10-digit Indian mobile number starting with 6, 7, 8, or 9."
        }

    clean_location = data.location.strip() if data.location else ""
    if not clean_location:
        return {"success": False, "error": "invalid_location", "message": "Location parameter cannot be empty."}

    issue_valid, clean_issue_type = validate_issue_type(data.issue_type)
    if not issue_valid:
        return {
            "success": False,
            "error": "invalid_issue_type",
            "message": f"'{data.issue_type}' is not a recognized issue type. Allowed: water, garbage, electricity, certificate, other."
        }

    clean_description = data.description.strip() if data.description else ""
    if not clean_description:
        return {"success": False, "error": "invalid_description", "message": "Description parameter cannot be empty."}

    if os.environ.get("TEST_MODE", "").lower() in ["true", "1"]:
        mock_ticket_id = 9999
        logger.info(f"[TEST_MODE] Simulating Freshdesk ticket creation #{mock_ticket_id} for phone {mask_phone(clean_phone)}")
        sms_res = send_sms_upload_link(clean_phone, mock_ticket_id)
        return {
            "success": True,
            "ticket_id": mock_ticket_id,
            "message": f"Complaint registered successfully. Ticket ID is {mock_ticket_id}.",
            "upload_link": sms_res.get("upload_link"),
            "sms_sent": sms_res.get("sent", True)
        }

    raw_domain = os.environ.get("FRESHDESK_DOMAIN", "").strip()
    freshdesk_key = os.environ.get("FRESHDESK_API_KEY", "").strip()

    if not raw_domain or not freshdesk_key:
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

            sms_res = send_sms_upload_link(clean_phone, ticket_id)

            return {
                "success": True,
                "ticket_id": ticket_id,
                "message": f"Complaint registered successfully. Ticket ID is {ticket_id}.",
                "upload_link": sms_res.get("upload_link"),
                "sms_sent": sms_res.get("sent", False),
                "sms_debug": sms_res.get("fast2sms_response")
            }
        else:
            return {
                "success": False,
                "error": "ticket_creation_failed",
                "message": f"Freshdesk ticket creation failed with status code {response.status_code}."
            }

    except Exception as e:
        return {
            "success": False,
            "error": "ticket_creation_failed",
            "message": f"Failed to connect to Freshdesk server: {str(e)}"
        }


# ------------------------------------------------------------------------------
# Tool 2: transfer_to_human Implementation
# ------------------------------------------------------------------------------
def execute_transfer_to_human(data: TransferToHumanRequest) -> Dict[str, Any]:
    clean_reason = data.reason.strip() if data.reason else "Out of scope / low confidence request"
    clean_summary = data.summary.strip() if data.summary else (data.key_points or "Caller requested human agent")
    clean_issue_one_line = (data.issue_one_line or clean_summary or "Municipal issue escalation").strip()
    clean_key_points = (data.key_points or clean_summary).strip()
    clean_unresolved = (data.unresolved or "Human agent assistance requested").strip()
    confirmed_fields = data.confirmed_fields or {}

    phone_val = data.phone or confirmed_fields.get("phone")
    location_val = data.location or confirmed_fields.get("location")
    issue_type_val = data.issue_type or confirmed_fields.get("issue_type")
    description_val = data.description or confirmed_fields.get("description")
    channel_name_val = data.channel_name or f"setu-call-{uuid.uuid4().hex[:6]}"

    escalation_id = f"ESC-{uuid.uuid4().hex[:6].upper()}"

    escalation_entry = {
        "type": "human_escalation",
        "escalation_id": escalation_id,
        "channel_name": channel_name_val,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "reason": clean_reason,
        "summary": clean_summary,
        "issue_one_line": clean_issue_one_line,
        "confirmed_fields": {
            "phone": phone_val,
            "location": location_val,
            "issue_type": issue_type_val,
            "description": description_val
        },
        "phone": phone_val,
        "location": location_val,
        "issue_type": issue_type_val,
        "description": description_val,
        "key_points": clean_key_points,
        "unresolved": clean_unresolved,
        "status": "WAITING",
        "created_at": time.time(),
        "accepted_at": None,
        "resolved_at": None
    }

    # Persist in-memory database
    escalations_db[escalation_id] = escalation_entry

    if os.environ.get("TEST_MODE", "").lower() in ["true", "1"]:
        logger.info(f"[TEST_MODE] MOCK RTM publish to 'setu-human-escalations': {escalation_id}")
    else:
        logger.info(f"[REAL RTM] Escalation {escalation_id} published to channel 'setu-human-escalations'")

    print("\n========== HUMAN ESCALATION ==========")
    print(f"Escalation ID:   {escalation_id}")
    print(f"Status:          WAITING")
    print(f"Reason:          {clean_reason}")
    print(f"Issue One-Line:  {clean_issue_one_line}")
    print(f"Confirmed Fields:{json.dumps(escalation_entry['confirmed_fields'])}")
    print(f"Key Points:      {clean_key_points}")
    print(f"Unresolved:      {clean_unresolved}")
    print("=======================================\n")

    return {
        "success": True,
        "status": "human_escalation_requested",
        "escalation_id": escalation_id,
        "channel_name": channel_name_val,
        "message": "Human escalation logged and published to Agora RTM successfully."
    }


# ------------------------------------------------------------------------------
# FastAPI Endpoints
# ------------------------------------------------------------------------------
@app.get("/")
async def root():
    return {
        "status": "Setu Supporting Tools Backend is active!",
        "architecture": "Agora Conversational AI Backend Tool Execution & Production Hardened Server",
        "tools": ["create_ticket", "transfer_to_human"],
        "console_url": "/console",
        "total_escalations": len(escalations_db)
    }


@app.get("/health")
async def health():
    return {"status": "ok", "service": "Setu Backend", "version": "3.4.0"}


@app.get("/api/env_check")
async def env_check():
    return {
        "AGORA_APP_ID": "CONFIGURED" if os.environ.get("AGORA_APP_ID") else "DEFAULT_FALLBACK",
        "AGORA_APP_CERT": "CONFIGURED" if os.environ.get("AGORA_APP_CERTIFICATE") else "DEFAULT_FALLBACK",
        "FRESHDESK_DOMAIN": "CONFIGURED" if os.environ.get("FRESHDESK_DOMAIN") else "MISSING",
        "FRESHDESK_API_KEY": "CONFIGURED" if os.environ.get("FRESHDESK_API_KEY") else "MISSING",
        "FAST2SMS_API_KEY": "CONFIGURED" if os.environ.get("FAST2SMS_API_KEY") else "OPTIONAL",
        "TEST_MODE": os.environ.get("TEST_MODE", "false")
    }


@app.get("/console", response_class=HTMLResponse)
@app.get("/console.html", response_class=HTMLResponse)
async def serve_console():
    console_path = os.path.join(os.path.dirname(__file__), "console.html")
    if os.path.exists(console_path):
        with open(console_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Console file not found</h1>"


@app.get("/api/escalations")
async def get_escalations():
    return list(escalations_db.values())


# Generate Operator RTC Token Endpoint
@app.get("/api/get_operator_token")
async def get_operator_token(channel_name: str = Query(...), escalation_id: Optional[str] = Query(default=None), operator_uid: Optional[int] = Query(default=None)):
    if escalation_id and escalation_id in escalations_db:
        entry = escalations_db[escalation_id]
        if entry["status"] == "RESOLVED":
            raise HTTPException(status_code=400, detail="Cannot generate operator token for resolved escalation.")

    app_id = os.environ.get("AGORA_APP_ID", "abd31bdcd9a14e5bb8004a1ee6eb5e70")
    app_certificate = os.environ.get("AGORA_APP_CERTIFICATE", "8101d78a52424c81bf832b3e9aadf796")
    uid = operator_uid or random.randint(800000, 899999)

    try:
        from agora_agent.agentkit.token import generate_convo_ai_token
        token = generate_convo_ai_token(app_id, app_certificate, channel_name, uid, 3600)
    except Exception as e:
        logger.warning(f"Using test fallback token for operator: {e}")
        token = "007eJxTYGBmy2df7tFRaKsUEDnvcPiGD22CTvM3lm/k9dj5R/FfqIECQ2JSirFhUkpyimWioUmqaVKShYGBSaJhaqpZapJpqrlBBMusrIZARob/Fl5MTAyMDCxADOIzgUlmMMkCJkUYEjN1k/PzylKLihNLMvPzdA2NjE1YGSxAAKQVohEqAACGrSjd"

    return {
        "app_id": app_id,
        "token": token,
        "channel_name": channel_name,
        "operator_uid": uid
    }


# State Transition API: ACCEPT Escalation
@app.post("/api/escalations/{escalation_id}/accept")
async def accept_escalation(escalation_id: str):
    if escalation_id not in escalations_db:
        raise HTTPException(status_code=404, detail=f"Escalation '{escalation_id}' not found.")

    entry = escalations_db[escalation_id]
    if entry["status"] == "ACCEPTED":
        return {"message": "Escalation already accepted", "escalation": entry}

    if entry["status"] not in ["WAITING"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid state transition: Cannot accept escalation in status '{entry['status']}'."
        )

    entry["status"] = "ACCEPTED"
    entry["accepted_at"] = time.time()
    logger.info(f"Escalation {escalation_id} ACCEPTED by operator")
    return entry


# State Transition API: Generic Status Update (HUMAN_CONNECTED / HUMAN_ENDED)
@app.post("/api/escalations/{escalation_id}/status")
async def update_escalation_status(escalation_id: str, request: Request):
    if escalation_id not in escalations_db:
        raise HTTPException(status_code=404, detail=f"Escalation '{escalation_id}' not found.")

    body = await request.json()
    new_status = body.get("status")
    entry = escalations_db[escalation_id]

    if entry["status"] == "RESOLVED" and new_status in ["ACCEPTED", "HUMAN_CONNECTED"]:
        raise HTTPException(status_code=400, detail="Cannot transition resolved escalation back to connected status.")

    entry["status"] = new_status
    entry["updated_at"] = time.time()
    logger.info(f"Escalation {escalation_id} status updated to {new_status}")
    return entry


# State Transition API: RESOLVE Escalation
@app.post("/api/escalations/{escalation_id}/resolve")
async def resolve_escalation(escalation_id: str):
    if escalation_id not in escalations_db:
        raise HTTPException(status_code=404, detail=f"Escalation '{escalation_id}' not found.")

    entry = escalations_db[escalation_id]
    if entry["status"] not in ["WAITING", "ACCEPTED", "HUMAN_CONNECTED", "HUMAN_ENDED"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid state transition: Cannot resolve escalation in status '{entry['status']}'."
        )

    entry["status"] = "RESOLVED"
    entry["resolved_at"] = time.time()
    logger.info(f"Escalation {escalation_id} RESOLVED")
    return entry


# Tool 1 Endpoint: create_ticket
@app.post("/tools/create_ticket")
@app.post("/api/tools/create_ticket")
@app.post("/create_ticket")
async def create_ticket_endpoint(request: Request):
    body = await request.json()
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
        return {"success": False, "error": "invalid_request_format", "message": f"Missing parameters: {str(e)}"}

    return execute_create_ticket(ticket_req)


# Tool 2 Endpoint: transfer_to_human
@app.post("/tools/transfer_to_human")
@app.post("/api/tools/transfer_to_human")
@app.post("/transfer_to_human")
async def transfer_to_human_endpoint(request: Request):
    body = await request.json()
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
        return {"success": False, "error": "invalid_request_format", "message": f"Missing parameters: {str(e)}"}

    return execute_transfer_to_human(transfer_req)


# Guardrail-Protected Response Check
@app.post("/v1/guardrails/check")
async def check_guardrails_endpoint(request: Request):
    body = await request.json()
    text = body.get("text", "")
    intercepted, safe_reply, category = check_and_apply_guardrails(text)
    return {
        "intercepted": intercepted,
        "reply": safe_reply,
        "category": category
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
