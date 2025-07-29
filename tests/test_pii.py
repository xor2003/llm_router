from app.pii import PIIScrubber


def test_pii_scrubber_scrubs_emails():
    scrubber = PIIScrubber()
    payload = {
        "messages": [
            {"role": "user", "content": "My email is test@example.com"},
            {"role": "assistant", "content": "Contact me at user@domain.org"},
        ],
    }
    scrubbed_payload, pii_map = scrubber.scrub(payload)

    assert "test@example.com" not in scrubbed_payload["messages"][0]["content"]
    assert "user@domain.org" not in scrubbed_payload["messages"][1]["content"]
    assert any("EMAIL" in key for key in pii_map.keys())


def test_pii_scrubber_scrubs_phones():
    scrubber = PIIScrubber()
    payload = {
        "messages": [
            {"role": "user", "content": "Call me at 555-123-4567"},
            {"role": "assistant", "content": "My number is (555) 987 6543"},
        ],
    }
    scrubbed_payload, pii_map = scrubber.scrub(payload)

    assert "555-123-4567" not in scrubbed_payload["messages"][0]["content"]
    assert "(555) 987 6543" not in scrubbed_payload["messages"][1]["content"]
    assert any("PHONE" in key for key in pii_map.keys())


def test_pii_scrubber_scrubs_credit_cards():
    scrubber = PIIScrubber()
    payload = {
        "messages": [
            {"role": "user", "content": "Card: 4111 1111 1111 1111"},
            {"role": "assistant", "content": "Use 5500 0000 0000 0004"},
        ],
    }
    scrubbed_payload, pii_map = scrubber.scrub(payload)

    assert "4111 1111 1111 1111" not in scrubbed_payload["messages"][0]["content"]
    assert "5500 0000 0000 0004" not in scrubbed_payload["messages"][1]["content"]
    assert any("CREDIT_CARD" in key for key in pii_map.keys())


def test_pii_scrubber_scrubs_ip_addresses():
    scrubber = PIIScrubber()
    payload = {
        "messages": [
            {"role": "user", "content": "Server IP is 192.168.1.1"},
            {"role": "assistant", "content": "Gateway at 10.0.0.1"},
        ],
    }
    scrubbed_payload, pii_map = scrubber.scrub(payload)

    assert "192.168.1.1" not in scrubbed_payload["messages"][0]["content"]
    assert "10.0.0.1" not in scrubbed_payload["messages"][1]["content"]
    assert any("IP_ADDRESS" in key for key in pii_map.keys())


def test_pii_scrubber_restores_pii():
    scrubber = PIIScrubber()
    original_content = "My email is test@example.com and phone is 555-123-4567"
    scrubbed_content, pii_map = scrubber._scrub_text(original_content)

    restored_content = scrubbed_content
    for placeholder, value in pii_map.items():
        restored_content = restored_content.replace(placeholder, value)

    assert restored_content == original_content


def test_pii_scrubber_with_custom_patterns():
    custom_patterns = {"SECRET_CODE": r"CODE-\d{4}"}
    scrubber = PIIScrubber(custom_patterns)
    payload = {"messages": [{"role": "user", "content": "Use code CODE-1234"}]}
    scrubbed_payload, pii_map = scrubber.scrub(payload)

    assert "CODE-1234" not in scrubbed_payload["messages"][0]["content"]
    assert any("SECRET_CODE" in key for key in pii_map.keys())


def test_pii_scrubber_handles_complex_content():
    scrubber = PIIScrubber()
    payload = {
        "messages": [
            {"role": "user", "content": "Contact me at test@example.com or 555-123-4567. Card: 4111 1111 1111 1111"},
        ],
    }
    scrubbed_payload, pii_map = scrubber.scrub(payload)

    content = scrubbed_payload["messages"][0]["content"]
    assert "test@example.com" not in content
    assert "555-123-4567" not in content
    assert "4111 1111 1111 1111" not in content
    assert len(pii_map) == 3


def test_pii_scrubber_restore_response():
    scrubber = PIIScrubber()
    original_content = "My email is test@example.com and phone is 555-123-4567"
    scrubbed_content, pii_map = scrubber._scrub_text(original_content)

    # Create a mock response payload
    response_payload = {"choices": [{"message": {"content": scrubbed_content}}]}

    # Restore the PII in the response
    restored_payload = scrubber.restore(response_payload, pii_map)
    restored_content = restored_payload["choices"][0]["message"]["content"]

    assert restored_content == original_content
