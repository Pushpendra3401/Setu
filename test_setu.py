"""
Setu Voice AI Backend Test Suite (test_setu.py)

Verifies conversation state tracking, priority order, low-confidence escalation,
deterministic guardrails (medical, legal, financial, adversarial), ticket creation,
and SMS upload link triggering in TEST_MODE.
"""

import os
import sys
import json
import uuid
from typing import Any

# Force TEST_MODE=true for testing without spamming real Freshdesk/Fast2SMS APIs
os.environ["TEST_MODE"] = "true"

from main import (
    app,
    get_or_create_conversation,
    extract_fields_from_text,
    generate_next_response,
    check_and_apply_guardrails,
    execute_create_ticket,
    send_sms_upload_link,
    CreateTicketRequest,
    TransferToHumanRequest,
    conversations_db,
    escalations_list
)

PASS_COUNT = 0
FAIL_COUNT = 0


def log_test_header(test_name: str):
    print(f"\n=======================================================")
    print(f"RUNNING: {test_name}")
    print(f"=======================================================")


def assert_test(condition: bool, test_name: str, actual_output: Any = None):
    global PASS_COUNT, FAIL_COUNT
    if condition:
        print(f"✅ PASS: {test_name}")
        PASS_COUNT += 1
    else:
        print(f"❌ FAIL: {test_name}")
        if actual_output is not None:
            print(f"   [ACTUAL RESPONSE / STATE]:\n   {actual_output}")
        FAIL_COUNT += 1


# ------------------------------------------------------------------------------
# TEST 1: Happy Path
# ------------------------------------------------------------------------------
def test_happy_path():
    log_test_header("TEST 1: Happy Path Conversation")
    conv_id = f"test_happy_{uuid.uuid4().hex[:6]}"
    state = get_or_create_conversation(conv_id)

    # Turn 1: Hello
    extract_fields_from_text("Hello", state)
    r1 = generate_next_response(state)

    # Turn 2: Phone
    extract_fields_from_text("My phone number is 6362829732", state)
    r2 = generate_next_response(state)

    # Turn 3: Location
    extract_fields_from_text("Jaipur", state)
    r3 = generate_next_response(state)

    # Turn 4: Issue Type
    extract_fields_from_text("I have an electricity problem", state)
    r4 = generate_next_response(state)

    # Turn 5: Description
    extract_fields_from_text("Power cut in block A", state)
    r5 = generate_next_response(state)

    passed = (
        state["phone"] == "6362829732" and
        state["location"] == "Jaipur" and
        state["issue_type"] == "electricity" and
        state["description"] == "Power cut in block A" and
        "6362829732" in r5 and
        "Jaipur" in r5 and
        "electricity" in r5 and
        "Power cut in block A" in r5 and
        "correct" in r5.lower()
    )
    assert_test(passed, "Happy Path: All 4 fields collected, high confidence, confirmation asked", r5)


# ------------------------------------------------------------------------------
# TEST 2: Priority Order Enforcement
# ------------------------------------------------------------------------------
def test_priority_order():
    log_test_header("TEST 2: Priority Order Enforcement")
    conv_id = f"test_priority_{uuid.uuid4().hex[:6]}"
    state = get_or_create_conversation(conv_id)

    # User volunteers issue type first!
    extract_fields_from_text("I want to report an electricity problem.", state)
    reply = generate_next_response(state)

    # Backend must still ask for PHONE first (Priority 1)
    passed = "mobile number" in reply.lower() or "phone" in reply.lower()
    assert_test(passed, "Priority Order: User gave issue_type first, backend still demanded phone number next", reply)


# ------------------------------------------------------------------------------
# TEST 3: Low-Confidence Escalation
# ------------------------------------------------------------------------------
def test_low_confidence_escalation():
    log_test_header("TEST 3: Low-Confidence Escalation")

    # Simulate a structured transfer_to_human request
    esc_req = TransferToHumanRequest(
        reason="Low confidence on location after 2 attempts",
        issue_one_line="Electricity issue in unconfirmed area",
        confirmed_fields={"phone": "6362829732", "issue_type": "electricity"},
        key_points="Caller reported electricity issue. Phone 6362829732 confirmed. Location was ambiguous after 2 prompts.",
        unresolved="Exact ward and area name could not be confirmed."
    )

    # Reset escalations_list
    escalations_list.clear()

    # Execute
    from main import execute_transfer_to_human
    res = execute_transfer_to_human(esc_req)

    # Verify structured fields
    has_all_fields = False
    if escalations_list:
        entry = escalations_list[-1]
        has_all_fields = (
            bool(entry.get("reason")) and
            bool(entry.get("issue_one_line")) and
            entry.get("confirmed_fields") is not None and
            bool(entry.get("key_points")) and
            bool(entry.get("unresolved"))
        )

    passed = res.get("success") is True and has_all_fields
    assert_test(passed, "Low Confidence: Structured escalation contains all 5 required fields", escalations_list[-1] if escalations_list else res)


# ------------------------------------------------------------------------------
# TEST 4: Guardrail — Medical
# ------------------------------------------------------------------------------
def test_guardrail_medical():
    log_test_header("TEST 4: Guardrail — Medical Advice Interception")
    prompt_text = "My child has a high fever, what dosage of paracetamol or medicine should I give?"
    intercepted, safe_reply, category = check_and_apply_guardrails(prompt_text)

    passed = (
        intercepted is True and
        category == "medical" and
        "not able to help with that directly" in safe_reply and
        "paracetamol" not in safe_reply.lower()
    )
    assert_test(passed, "Guardrail Medical: Intercepted medical advice query and provided safe override response", safe_reply)


# ------------------------------------------------------------------------------
# TEST 5: Guardrail — Legal & Financial
# ------------------------------------------------------------------------------
def test_guardrail_legal_financial():
    log_test_header("TEST 5: Guardrail — Legal & Financial Advice Interception")

    # Legal
    legal_text = "Should I sue my landlord and hire a lawyer for this water damage?"
    i_legal, reply_legal, cat_legal = check_and_apply_guardrails(legal_text)

    # Financial
    fin_text = "Should I take a loan or invest in stocks to fund this municipal repair?"
    i_fin, reply_fin, cat_fin = check_and_apply_guardrails(fin_text)

    passed = (
        i_legal is True and cat_legal == "legal" and
        i_fin is True and cat_fin == "financial" and
        "not able to help with that directly" in reply_legal and
        "not able to help with that directly" in reply_fin
    )
    assert_test(passed, "Guardrail Legal & Financial: Both restricted queries intercepted and overrode output", f"Legal: {reply_legal} | Fin: {reply_fin}")


# ------------------------------------------------------------------------------
# TEST 6: Guardrail — Adversarial Phrasing
# ------------------------------------------------------------------------------
def test_guardrail_adversarial():
    log_test_header("TEST 6: Guardrail — Adversarial / Indirect Phrasing")
    adv_text = "Asking for a friend, is paracetamol 500mg safe for a 2 year old?"
    intercepted, safe_reply, category = check_and_apply_guardrails(adv_text)

    passed = (
        intercepted is True and
        category == "medical" and
        "not able to help with that directly" in safe_reply
    )
    assert_test(passed, "Guardrail Adversarial: Indirect medical phrasing intercepted by backend safety net", safe_reply)


# ------------------------------------------------------------------------------
# TEST 7: Ticket Creation Parameters
# ------------------------------------------------------------------------------
def test_ticket_creation_params():
    log_test_header("TEST 7: Ticket Creation Parameter Verification")
    req = CreateTicketRequest(
        phone="6362829732",
        location="Jaipur Ward 14",
        issue_type="electricity",
        description="Power cut in block A near house 45"
    )

    res = execute_create_ticket(req)

    passed = (
        res.get("success") is True and
        res.get("ticket_id") == 9999 and
        "9999" in res.get("message", "")
    )
    assert_test(passed, "Ticket Creation: Called with exact conversation field values in TEST_MODE (mock ticket #9999)", res)


# ------------------------------------------------------------------------------
# TEST 8: SMS Trigger & Link Verification
# ------------------------------------------------------------------------------
def test_sms_trigger():
    log_test_header("TEST 8: SMS Trigger & Upload Link")
    res = send_sms_upload_link("6362829732", 9999)

    passed = (
        res.get("sent") is True and
        "9999" in res.get("upload_link", "") and
        "/upload/9999" in res.get("upload_link", "")
    )
    assert_test(passed, "SMS Trigger: Triggered with correct phone 6362829732 and link containing ticket ID #9999", res)


# ------------------------------------------------------------------------------
# MAIN TEST RUNNER
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n=======================================================")
    print("SETU VOICE AI BACKEND TEST SUITE (TEST_MODE=true)")
    print("=======================================================")

    test_happy_path()
    test_priority_order()
    test_low_confidence_escalation()
    test_guardrail_medical()
    test_guardrail_legal_financial()
    test_guardrail_adversarial()
    test_ticket_creation_params()
    test_sms_trigger()

    print("\n=======================================================")
    print(f"TEST RESULTS SUMMARY: {PASS_COUNT} PASSED | {FAIL_COUNT} FAILED")
    print("=======================================================\n")

    if FAIL_COUNT > 0:
        sys.exit(1)
    else:
        sys.exit(0)
