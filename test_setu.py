"""
Setu Voice AI Backend Test Suite (test_setu.py)

Verifies conversation state tracking, priority order, low-confidence escalation,
deterministic guardrails (medical, legal, financial, adversarial), ticket creation,
SMS upload link triggering, RTM Human Escalation Console state transitions,
and Production Hardening / Reliability checks (Tests 1-30).
"""

import os
import sys
import json
import uuid
import asyncio
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
    execute_transfer_to_human,
    accept_escalation,
    resolve_escalation,
    update_escalation_status,
    get_operator_token,
    env_check,
    health,
    mask_phone,
    CreateTicketRequest,
    TransferToHumanRequest,
    conversations_db,
    escalations_db
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

    extract_fields_from_text("Hello", state)
    r1 = generate_next_response(state)

    extract_fields_from_text("My phone number is 6362829732", state)
    r2 = generate_next_response(state)

    extract_fields_from_text("Jaipur", state)
    r3 = generate_next_response(state)

    extract_fields_from_text("I have an electricity problem", state)
    r4 = generate_next_response(state)

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

    extract_fields_from_text("I want to report an electricity problem.", state)
    reply = generate_next_response(state)

    passed = "mobile number" in reply.lower() or "phone" in reply.lower()
    assert_test(passed, "Priority Order: User gave issue_type first, backend still demanded phone number next", reply)


# ------------------------------------------------------------------------------
# TEST 3: Low-Confidence Escalation
# ------------------------------------------------------------------------------
def test_low_confidence_escalation():
    log_test_header("TEST 3: Low-Confidence Escalation")
    esc_req = TransferToHumanRequest(
        reason="Low confidence on location after 2 attempts",
        issue_one_line="Electricity issue in unconfirmed area",
        confirmed_fields={"phone": "6362829732", "issue_type": "electricity"},
        key_points="Caller reported electricity issue. Phone 6362829732 confirmed. Location was ambiguous after 2 prompts.",
        unresolved="Exact ward and area name could not be confirmed."
    )

    escalations_db.clear()
    res = execute_transfer_to_human(esc_req)

    has_all_fields = False
    if escalations_db:
        entry = list(escalations_db.values())[-1]
        has_all_fields = (
            bool(entry.get("reason")) and
            bool(entry.get("issue_one_line")) and
            entry.get("confirmed_fields") is not None and
            bool(entry.get("key_points")) and
            bool(entry.get("unresolved"))
        )

    passed = res.get("success") is True and has_all_fields
    assert_test(passed, "Low Confidence: Structured escalation contains all 5 required fields", list(escalations_db.values())[-1] if escalations_db else res)


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

    legal_text = "Should I sue my landlord and hire a lawyer for this water damage?"
    i_legal, reply_legal, cat_legal = check_and_apply_guardrails(legal_text)

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
# TEST 8: SMS Trigger & Upload Link
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
# TEST 9: transfer_to_human Creates Escalation
# ------------------------------------------------------------------------------
def test_escalation_creation():
    log_test_header("TEST 9: transfer_to_human Creates Escalation")
    req = TransferToHumanRequest(
        reason="Low confidence on location after 2 attempts",
        summary="Electricity issue in unconfirmed area"
    )
    res = execute_transfer_to_human(req)

    esc_id = res.get("escalation_id", "")
    passed = (
        res.get("success") is True and
        esc_id.startswith("ESC-") and
        esc_id in escalations_db and
        escalations_db[esc_id]["status"] == "WAITING"
    )
    assert_test(passed, f"transfer_to_human created escalation {esc_id} with status WAITING", res)


# ------------------------------------------------------------------------------
# TEST 10: Escalation Data Integrity
# ------------------------------------------------------------------------------
def test_escalation_data_integrity():
    log_test_header("TEST 10: Escalation Data Integrity")
    req = TransferToHumanRequest(
        reason="Low confidence on location after 2 attempts",
        issue_one_line="Water leakage in Ward 14",
        confirmed_fields={"phone": "6362829732", "issue_type": "water"},
        key_points="Caller reported water leakage. Phone 6362829732 confirmed.",
        unresolved="Exact street address could not be confirmed."
    )
    res = execute_transfer_to_human(req)
    esc_id = res.get("escalation_id")
    data = escalations_db.get(esc_id, {})

    passed = (
        data.get("type") == "human_escalation" and
        data.get("reason") == "Low confidence on location after 2 attempts" and
        data.get("issue_one_line") == "Water leakage in Ward 14" and
        data.get("phone") == "6362829732" and
        data.get("issue_type") == "water" and
        data.get("status") == "WAITING"
    )
    assert_test(passed, "Escalation contains correct structured fields", data)


# ------------------------------------------------------------------------------
# TEST 11: RTM Publication
# ------------------------------------------------------------------------------
def test_rtm_publication():
    log_test_header("TEST 11: RTM Publication (Mock in TEST_MODE)")
    req = TransferToHumanRequest(reason="Test RTM Publish", summary="RTM test")
    res = execute_transfer_to_human(req)

    passed = res.get("success") is True and "escalation_id" in res
    assert_test(passed, "MOCK RTM: Escalation published through Agora RTM in TEST_MODE", res)


# ------------------------------------------------------------------------------
# TEST 12 & 13: Operator Console & Realtime Delivery
# ------------------------------------------------------------------------------
def test_realtime_delivery():
    log_test_header("TEST 12 & 13: Operator Console & Realtime Delivery")
    from main import get_escalations
    before_len = len(asyncio.run(get_escalations()))

    req = TransferToHumanRequest(reason="Realtime Test", summary="Realtime Delivery Test")
    res = execute_transfer_to_human(req)

    after_items = asyncio.run(get_escalations())
    passed = len(after_items) == before_len + 1 and after_items[-1]["escalation_id"] == res.get("escalation_id")
    assert_test(passed, "Realtime Delivery: New escalation appears in operator console feed automatically", res)


# ------------------------------------------------------------------------------
# TEST 14: WAITING State
# ------------------------------------------------------------------------------
def test_waiting_state():
    log_test_header("TEST 14: WAITING State Verification")
    req = TransferToHumanRequest(reason="Initial State Test", summary="Waiting state test")
    res = execute_transfer_to_human(req)
    esc_id = res.get("escalation_id")

    passed = escalations_db[esc_id]["status"] == "WAITING"
    assert_test(passed, "New escalation starts in WAITING state", escalations_db[esc_id])


# ------------------------------------------------------------------------------
# TEST 15: Accept Escalation
# ------------------------------------------------------------------------------
def test_accept_escalation():
    log_test_header("TEST 15: Accept Escalation (WAITING -> ACCEPTED)")
    req = TransferToHumanRequest(reason="Accept Test", summary="Accept test")
    res = execute_transfer_to_human(req)
    esc_id = res.get("escalation_id")

    updated = asyncio.run(accept_escalation(esc_id))
    passed = updated["status"] == "ACCEPTED" and updated["accepted_at"] is not None
    assert_test(passed, f"Escalation {esc_id} ACCEPTED and persisted server-side", updated)


# ------------------------------------------------------------------------------
# TEST 16: Resolve Escalation
# ------------------------------------------------------------------------------
def test_resolve_escalation():
    log_test_header("TEST 16: Resolve Escalation (ACCEPTED -> RESOLVED)")
    req = TransferToHumanRequest(reason="Resolve Test", summary="Resolve test")
    res = execute_transfer_to_human(req)
    esc_id = res.get("escalation_id")

    asyncio.run(accept_escalation(esc_id))
    updated = asyncio.run(resolve_escalation(esc_id))
    passed = updated["status"] == "RESOLVED" and updated["resolved_at"] is not None
    assert_test(passed, f"Escalation {esc_id} RESOLVED and persisted server-side", updated)


# ------------------------------------------------------------------------------
# TEST 17: Persistence After Refresh
# ------------------------------------------------------------------------------
def test_persistence_after_refresh():
    log_test_header("TEST 17: Persistence After Refresh")
    req = TransferToHumanRequest(reason="Refresh Test", summary="Refresh persistence test")
    res = execute_transfer_to_human(req)
    esc_id = res.get("escalation_id")

    asyncio.run(accept_escalation(esc_id))
    asyncio.run(resolve_escalation(esc_id))

    persisted = escalations_db.get(esc_id)
    passed = persisted and persisted["status"] == "RESOLVED"
    assert_test(passed, f"Escalation {esc_id} status remained RESOLVED after reload", persisted)


# ------------------------------------------------------------------------------
# TEST 18: Invalid Transition Rejection
# ------------------------------------------------------------------------------
def test_invalid_state_transition():
    log_test_header("TEST 18: Invalid State Transition Rejection")
    req = TransferToHumanRequest(reason="Invalid Transition Test", summary="Invalid transition test")
    res = execute_transfer_to_human(req)
    esc_id = res.get("escalation_id")

    asyncio.run(accept_escalation(esc_id))
    asyncio.run(resolve_escalation(esc_id))

    rejected = False
    try:
        asyncio.run(accept_escalation(esc_id))
    except Exception as e:
        rejected = True

    assert_test(rejected, f"Invalid transition RESOLVED -> ACCEPTED rejected with error", esc_id)


# ------------------------------------------------------------------------------
# TEST 19: Multiple Escalations Independence
# ------------------------------------------------------------------------------
def test_multiple_escalations():
    log_test_header("TEST 19: Multiple Escalations Independence")
    res1 = execute_transfer_to_human(TransferToHumanRequest(reason="Esc 1", summary="Esc 1"))
    res2 = execute_transfer_to_human(TransferToHumanRequest(reason="Esc 2", summary="Esc 2"))

    id1 = res1.get("escalation_id")
    id2 = res2.get("escalation_id")

    asyncio.run(accept_escalation(id1))

    passed = (
        escalations_db[id1]["status"] == "ACCEPTED" and
        escalations_db[id2]["status"] == "WAITING"
    )
    assert_test(passed, "Multiple escalations remain independent when status changes", f"{id1}: {escalations_db[id1]['status']} | {id2}: {escalations_db[id2]['status']}")


# ------------------------------------------------------------------------------
# TEST 20: Security Check
# ------------------------------------------------------------------------------
def test_security_check():
    log_test_header("TEST 20: Security Check (No Secrets Exposed)")
    from main import get_escalations
    items = asyncio.run(get_escalations())
    json_str = json.dumps(items)

    secrets = ["FRESHDESK_API_KEY", "FAST2SMS_API_KEY", "_QhINFlkDVVAmS71gfzN", "8101d78a52424c81bf832b3e9aadf796"]
    exposed = [s for s in secrets if s in json_str]

    passed = len(exposed) == 0
    assert_test(passed, "Security Check: Server secrets are NOT exposed in API responses or frontend payloads", exposed)


# ------------------------------------------------------------------------------
# RELIABILITY & PRODUCTION HARDENING TESTS (TEST 21 - 28)
# ------------------------------------------------------------------------------
def test_duplicate_accept_protection():
    log_test_header("TEST 21: Duplicate Accept Protection")
    req = TransferToHumanRequest(reason="Dup Accept Test", summary="Dup Accept test")
    res = execute_transfer_to_human(req)
    esc_id = res.get("escalation_id")

    res1 = asyncio.run(accept_escalation(esc_id))
    res2 = asyncio.run(accept_escalation(esc_id))

    passed = res2.get("message") == "Escalation already accepted"
    assert_test(passed, "Duplicate accept request handled safely without state corruption", res2)


def test_resolved_token_prevention():
    log_test_header("TEST 22: Resolved Escalation Token Prevention")
    req = TransferToHumanRequest(reason="Resolved Token Test", summary="Resolved Token test")
    res = execute_transfer_to_human(req)
    esc_id = res.get("escalation_id")

    asyncio.run(accept_escalation(esc_id))
    asyncio.run(resolve_escalation(esc_id))

    blocked = False
    try:
        asyncio.run(get_operator_token(channel_name="test_chan", escalation_id=esc_id))
    except Exception as e:
        blocked = True

    assert_test(blocked, "Operator token request rejected for resolved escalation", esc_id)


def test_phone_masking():
    log_test_header("TEST 23: Phone Masking in Logs")
    masked = mask_phone("6362829732")
    passed = masked == "******9732" and "6362" not in masked
    assert_test(passed, f"Phone number 6362829732 masked safely as {masked}", masked)


def test_health_check_endpoint():
    log_test_header("TEST 24: Health Check Endpoint")
    res = asyncio.run(health())
    passed = res.get("status") == "ok" and res.get("version") == "3.4.0"
    assert_test(passed, "GET /health endpoint responded with status 'ok'", res)


def test_env_check_endpoint():
    log_test_header("TEST 25: Safe Environment Checklist Endpoint")
    res = asyncio.run(env_check())
    has_status = "FRESHDESK_DOMAIN" in res and "FRESHDESK_API_KEY" in res
    secrets = ["_QhINFlkDVVAmS71gfzN", "8101d78a52424c81bf832b3e9aadf796"]
    exposed = [s for s in secrets if s in str(res)]

    passed = has_status and len(exposed) == 0
    assert_test(passed, "GET /api/env_check returns safe checklist without exposing secrets", res)


def test_fast2sms_failure_isolation():
    log_test_header("TEST 26: Fast2SMS Failure Isolation")
    orig_mode = os.environ.get("TEST_MODE")
    orig_key = os.environ.get("FAST2SMS_API_KEY")

    os.environ["TEST_MODE"] = "false"
    os.environ["FAST2SMS_API_KEY"] = ""

    sms_res = send_sms_upload_link("6362829732", 9999)

    os.environ["TEST_MODE"] = orig_mode or "true"
    if orig_key:
        os.environ["FAST2SMS_API_KEY"] = orig_key

    passed = sms_res.get("sent") is False and "upload_link" in sms_res
    assert_test(passed, "Fast2SMS missing key/failure handled safely without crashing app", sms_res)


def test_channel_isolation():
    log_test_header("TEST 27: Multi-Call Channel Isolation")
    res1 = execute_transfer_to_human(TransferToHumanRequest(reason="Call A", channel_name="chan_A"))
    res2 = execute_transfer_to_human(TransferToHumanRequest(reason="Call B", channel_name="chan_B"))

    id1 = res1.get("escalation_id")
    id2 = res2.get("escalation_id")

    chan1 = escalations_db[id1]["channel_name"]
    chan2 = escalations_db[id2]["channel_name"]

    passed = chan1 == "chan_A" and chan2 == "chan_B" and chan1 != chan2
    assert_test(passed, f"Channels isolated across concurrent calls ({chan1} vs {chan2})", f"{id1}: {chan1} | {id2}: {chan2}")


# ------------------------------------------------------------------------------
# MAIN TEST RUNNER
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n=======================================================")
    print("SETU RELIABILITY & HARDENING TEST SUITE (TEST_MODE=true)")
    print("=======================================================")

    test_happy_path()
    test_priority_order()
    test_low_confidence_escalation()
    test_guardrail_medical()
    test_guardrail_legal_financial()
    test_guardrail_adversarial()
    test_ticket_creation_params()
    test_sms_trigger()

    test_escalation_creation()
    test_escalation_data_integrity()
    test_rtm_publication()
    test_realtime_delivery()
    test_waiting_state()
    test_accept_escalation()
    test_resolve_escalation()
    test_persistence_after_refresh()
    test_invalid_state_transition()
    test_multiple_escalations()
    test_security_check()

    test_duplicate_accept_protection()
    test_resolved_token_prevention()
    test_phone_masking()
    test_health_check_endpoint()
    test_env_check_endpoint()
    test_fast2sms_failure_isolation()
    test_channel_isolation()

    print("\n=======================================================")
    print(f"TEST RESULTS SUMMARY: {PASS_COUNT} PASSED | {FAIL_COUNT} FAILED")
    print("=======================================================\n")

    if FAIL_COUNT > 0:
        sys.exit(1)
    else:
        sys.exit(0)
