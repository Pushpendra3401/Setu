from fastapi import FastAPI, HTTPException, Request, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
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
    title="Setu Municipal Helpline Backend Tools & Safety Guardrails",
    description="Backend tool execution server with deterministic guardrails, create_ticket, Fast2SMS evidence link, and Freshdesk integration",
    version="3.1.0"
)

logger = logging.getLogger("uvicorn.error")

ALLOWED_ISSUE_TYPES = {"water", "garbage", "electricity", "certificate", "other"}

WORD_TO_DIGIT = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9"
}

# In-memory log of structured escalations for console.html
escalations_list: List[Dict[str, Any]] = []

# In-memory conversation state database
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
# Deterministic Backend Guardrails Definitions (Medical, Legal, Financial, Emergency)
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
    """
    Deterministically scans reply_text against medical, legal, financial, and emergency regex patterns.
    If detected, returns (True, SAFE_GUARDRAIL_RESPONSE, matched_category) and triggers transfer_to_human.
    """
    if not reply_text:
        return False, reply_text, None

    lowered = reply_text.lower()

    for category, patterns in ALL_GUARDRAIL_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, lowered):
                logger.warning(f"DETERMINISTIC GUARDRAIL INTERCEPTED [{category.upper()}]: '{reply_text}'")

                # Automatically trigger transfer_to_human escalation
                esc_req = TransferToHumanRequest(
                    reason=f"Guardrail Intercepted: {category.title()} Advice Detected",
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
    issue_one_line: str = Field(..., description="One line summary of the issue")
    confirmed_fields: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Dictionary of locked-in fields")
    key_points: str = Field(..., description="2-3 sentences max detailing what was discussed without filler")
    unresolved: str = Field(..., description="What is uncertain and specifically why a human agent is needed")


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

    # 1. Extract Phone Number
    is_valid_phone, clean_phone = validate_indian_phone(text)
    if is_valid_phone:
        state["phone"] = clean_phone
        state["phone_confidence"] = "high"
    elif len(clean_phone) > 0 and state["phone_confidence"] != "high":
        state["phone_confidence"] = "low"

    # 2. Extract Location (only if location is not already locked in with high confidence)
    if state["location_confidence"] != "high":
        location_match = re.search(r"\b(ward\s*\d+|jaipur|delhi|mumbai|bangalore|pune|sector\s*\d+)\b", lowered)
        if location_match:
            state["location"] = location_match.group(0).title()
            state["location_confidence"] = "high"
        elif state["phone_confidence"] == "high" and not is_valid_phone:
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
        logger.info(f"[TEST_MODE] Simulating SMS to {phone}. Upload link: {upload_link}")
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

    payload = {
        "route": "q",
        "message": sms_message,
        "language": "english",
        "flash": 0,
        "numbers": phone
    }
    headers = {
        "authorization": api_key,
        "Content-Type": "application/json"
    }

    try:
        res = requests.post(url, json=payload, headers=headers, timeout=10)
        res_data = res.json() if "json" in res.headers.get("content-type", "") else {"text": res.text}
        logger.info(f"Fast2SMS API Response for {phone}: {res_data}")

        is_success = res.status_code == 200 and res_data.get("return") is True
        return {
            "sent": is_success,
            "upload_link": upload_link,
            "fast2sms_response": res_data
        }
    except Exception as e:
        logger.exception(f"Fast2SMS request failed for phone {phone}")
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
        logger.info(f"[TEST_MODE] Simulating Freshdesk ticket creation #{mock_ticket_id} for phone {clean_phone}")
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
                "sms_sent": sms_res.get("sent", False)
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
    clean_reason = data.reason.strip() if data.reason else ""
    clean_issue_one_line = data.issue_one_line.strip() if data.issue_one_line else ""
    clean_key_points = data.key_points.strip() if data.key_points else ""
    clean_unresolved = data.unresolved.strip() if data.unresolved else ""
    confirmed_fields = data.confirmed_fields or {}

    if not clean_reason or not clean_issue_one_line or not clean_key_points or not clean_unresolved:
        return {"success": False, "error": "invalid_parameters", "message": "All structured parameters are required."}

    escalation_entry = {
        "id": f"esc-{int(time.time())}",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "reason": clean_reason,
        "issue_one_line": clean_issue_one_line,
        "confirmed_fields": confirmed_fields,
        "key_points": clean_key_points,
        "unresolved": clean_unresolved
    }

    escalations_list.append(escalation_entry)

    print("\n========== HUMAN ESCALATION ==========")
    print(f"Reason:          {clean_reason}")
    print(f"Issue One-Line:  {clean_issue_one_line}")
    print(f"Confirmed Fields:{json.dumps(confirmed_fields)}")
    print(f"Key Points:      {clean_key_points}")
    print(f"Unresolved:      {clean_unresolved}")
    print("=======================================\n")

    return {
        "success": True,
        "status": "human_escalation_requested",
        "message": "Human escalation logged and published successfully."
    }


# ------------------------------------------------------------------------------
# FastAPI Endpoints
# ------------------------------------------------------------------------------
@app.get("/")
async def root():
    return {
        "status": "Setu Supporting Tools Backend is active!",
        "architecture": "Agora Conversational AI Backend Tool Execution & Guardrail Server",
        "tools": ["create_ticket", "transfer_to_human"],
        "console_url": "/console"
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


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
    return escalations_list


@app.get("/upload/{ticket_id}", response_class=HTMLResponse)
async def upload_page(ticket_id: str):
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Setu — Upload Photo Evidence for Ticket #{ticket_id}</title>
  <style>
    :root {{
      --bg: #0f172a;
      --card: #1e293b;
      --primary: #3b82f6;
      --primary-hover: #2563eb;
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --success: #10b981;
      --border: #334155;
    }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background-color: var(--bg);
      color: var(--text);
      margin: 0;
      padding: 20px;
      display: flex;
      justify-content: center;
      align-items: center;
      min-height: 100vh;
      box-sizing: border-box;
    }}
    .upload-card {{
      background-color: var(--card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 24px;
      width: 100%;
      max-width: 420px;
      box-shadow: 0 10px 15px -3px rgba(0,0,0,0.3);
      text-align: center;
    }}
    .badge {{
      display: inline-block;
      background-color: rgba(59, 130, 246, 0.15);
      color: #60a5fa;
      border: 1px solid rgba(59, 130, 246, 0.3);
      padding: 4px 12px;
      border-radius: 20px;
      font-size: 13px;
      font-weight: 600;
      margin-bottom: 12px;
    }}
    h1 {{ font-size: 20px; margin: 0 0 8px 0; }}
    p {{ font-size: 14px; color: var(--text-muted); margin: 0 0 20px 0; }}
    .file-drop {{
      border: 2px dashed var(--border);
      border-radius: 8px;
      padding: 20px;
      margin-bottom: 20px;
      background: #0f172a;
      cursor: pointer;
    }}
    input[type="file"] {{ display: none; }}
    .file-btn {{
      background: var(--border);
      color: var(--text);
      padding: 10px 16px;
      border-radius: 6px;
      font-size: 14px;
      font-weight: 600;
      display: inline-block;
      margin-bottom: 10px;
    }}
    .preview {{
      max-width: 100%;
      max-height: 200px;
      border-radius: 6px;
      margin-top: 10px;
      display: none;
    }}
    .submit-btn {{
      background-color: var(--primary);
      color: #fff;
      border: none;
      width: 100%;
      padding: 12px;
      border-radius: 6px;
      font-size: 15px;
      font-weight: 600;
      cursor: pointer;
      transition: background 0.2s;
    }}
    .submit-btn:hover {{ background-color: var(--primary-hover); }}
    .submit-btn:disabled {{ opacity: 0.6; cursor: not-allowed; }}
    .success-screen {{ display: none; }}
    .success-icon {{ font-size: 48px; color: var(--success); margin-bottom: 12px; }}
  </style>
</head>
<body>

  <div class="upload-card">
    <div id="form-screen">
      <span class="badge">Ticket #{ticket_id}</span>
      <h1>Upload Photo Evidence</h1>
      <p>Attach a photo of the municipal issue to update your complaint ticket.</p>

      <form id="upload-form">
        <div class="file-drop" onclick="document.getElementById('photo-input').click()">
          <div class="file-btn">📷 Choose or Capture Photo</div>
          <div id="file-name" style="font-size: 12px; color: var(--text-muted);">No file selected</div>
          <img id="preview" class="preview" alt="Preview" />
        </div>
        <input type="file" id="photo-input" name="photo" accept="image/*" capture="environment" required>

        <button type="submit" id="submit-btn" class="submit-btn" disabled>Upload Photo Evidence</button>
      </form>
    </div>

    <div id="success-screen" class="success-screen">
      <div class="success-icon">✅</div>
      <h1>Photo Uploaded Successfully!</h1>
      <p>Your photo evidence has been attached to Freshdesk Ticket #{ticket_id}.</p>
    </div>
  </div>

  <script>
    const photoInput = document.getElementById('photo-input');
    const fileNameDiv = document.getElementById('file-name');
    const previewImg = document.getElementById('preview');
    const submitBtn = document.getElementById('submit-btn');
    const form = document.getElementById('upload-form');
    const formScreen = document.getElementById('form-screen');
    const successScreen = document.getElementById('success-screen');

    photoInput.addEventListener('change', (e) => {{
      const file = e.target.files[0];
      if (file) {{
        fileNameDiv.textContent = file.name;
        submitBtn.disabled = false;

        const reader = new FileReader();
        reader.onload = (event) => {{
          previewImg.src = event.target.result;
          previewImg.style.display = 'block';
        }};
        reader.readAsDataURL(file);
      }}
    }});

    form.addEventListener('submit', async (e) => {{
      e.preventDefault();
      const file = photoInput.files[0];
      if (!file) return;

      submitBtn.disabled = true;
      submitBtn.textContent = 'Uploading...';

      const formData = new FormData();
      formData.append('photo', file);

      try {{
        const res = await fetch('/upload/{ticket_id}', {{
          method: 'POST',
          body: formData
        }});

        if (res.ok) {{
          formScreen.style.display = 'none';
          successScreen.style.display = 'block';
        }} else {{
          const err = await res.json();
          alert('Upload failed: ' + (err.message || 'Unknown error'));
          submitBtn.disabled = false;
          submitBtn.textContent = 'Upload Photo Evidence';
        }}
      }} catch (error) {{
        alert('Network error while uploading photo.');
        submitBtn.disabled = false;
        submitBtn.textContent = 'Upload Photo Evidence';
      }}
    }});
  </script>
</body>
</html>"""


@app.post("/upload/{ticket_id}")
async def process_photo_upload(ticket_id: str, photo: UploadFile = File(...)):
    raw_domain = os.environ.get("FRESHDESK_DOMAIN", "").strip()
    freshdesk_key = os.environ.get("FRESHDESK_API_KEY", "").strip()

    if not raw_domain or not freshdesk_key:
        raise HTTPException(status_code=500, detail="Freshdesk credentials not configured on server.")

    freshdesk_domain = raw_domain.replace("https://", "").replace("http://", "").rstrip("/")
    if not freshdesk_domain.endswith(".freshdesk.com"):
        freshdesk_domain = f"{freshdesk_domain}.freshdesk.com"

    notes_url = f"https://{freshdesk_domain}/api/v2/tickets/{ticket_id}/notes"

    contents = await photo.read()
    filename = photo.filename or "photo_evidence.jpg"
    content_type = photo.content_type or "image/jpeg"

    files = [
        ("attachments[]", (filename, contents, content_type))
    ]
    data = {
        "body": f"Photo evidence uploaded by caller for Ticket #{ticket_id} via Setu SMS upload link."
    }

    try:
        res = requests.post(
            notes_url,
            data=data,
            files=files,
            auth=(freshdesk_key, "X"),
            timeout=15
        )

        if res.status_code == 201:
            logger.info(f"Successfully attached photo evidence to Freshdesk Ticket #{ticket_id}")
            return {
                "success": True,
                "ticket_id": ticket_id,
                "message": f"Photo evidence successfully attached to Freshdesk Ticket #{ticket_id}."
            }
        else:
            logger.error(f"Freshdesk note attachment error status {res.status_code}: {res.text}")
            raise HTTPException(status_code=400, detail=f"Freshdesk returned status code {res.status_code}")

    except Exception as e:
        logger.exception(f"Failed to upload attachment to Freshdesk Ticket #{ticket_id}")
        raise HTTPException(status_code=500, detail=f"Failed to attach photo to Freshdesk: {str(e)}")


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


# Guardrail-Protected Response Check (Backend Safety Net)
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


# Conversational Chat Completions Endpoint
@app.get("/v1/chat/completions")
async def chat_get():
    return {"message": "Setu Supporting Backend is active. Waiting for POST requests."}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    messages: List[Dict[str, Any]] = body.get("messages", [])
    conversation_id = body.get("conversation_id", body.get("channel_name", "default_demo"))
    stream = body.get("stream", True)

    state = get_or_create_conversation(conversation_id)

    user_messages = [m for m in messages if m.get("role") == "user"]
    if user_messages:
        latest_text = user_messages[-1].get("content", "")
        # Check guardrails first
        intercepted, safe_reply, category = check_and_apply_guardrails(latest_text)
        if intercepted:
            reply_text = safe_reply
        else:
            extract_fields_from_text(latest_text, state)
            reply_text = generate_next_response(state)
    else:
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
