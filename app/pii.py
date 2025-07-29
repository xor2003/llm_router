import re
from typing import Any, Dict, List, Tuple

class PIIScrubber:
    """
    A class to detect and substitute personally identifiable information (PII)
    in a given text.
    """

    def __init__(self, custom_patterns: Dict[str, str] | None = None):
        # Default patterns for common PII types
        default_patterns = {
            "EMAIL": r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
            "PHONE": r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}",
            "CREDIT_CARD": r"\b(?:\d[ -]*?){13,16}\b",
            "IP_ADDRESS": r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
            "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
        }
        
        # Merge custom patterns with defaults (custom takes precedence)
        self.pii_patterns = default_patterns.copy()
        if custom_patterns:
            self.pii_patterns.update(custom_patterns)

    def scrub(self, payload: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, str]]:
        """
        Scans the payload for PII and replaces it with placeholders.

        Returns a tuple containing the scrubbed payload and a dictionary
        mapping placeholders to the original PII.
        """
        scrubbed_payload = payload.copy()
        pii_map: Dict[str, str] = {}
        
        for message in scrubbed_payload.get("messages", []):
            if "content" in message and isinstance(message["content"], str):
                content, new_pii = self._scrub_text(message["content"])
                message["content"] = content
                pii_map.update(new_pii)

        return scrubbed_payload, pii_map

    def _scrub_text(self, text: str) -> Tuple[str, Dict[str, str]]:
        """
        Scans a string for PII and replaces it with placeholders.
        """
        pii_map: Dict[str, str] = {}
        for pii_type, pattern in self.pii_patterns.items():
            for match in re.finditer(pattern, text):
                pii_value = match.group(0)
                placeholder = f"[{pii_type}_{len(pii_map)}]"
                text = text.replace(pii_value, placeholder)
                pii_map[placeholder] = pii_value
        return text, pii_map

    def restore(self, payload: Dict[str, Any], pii_map: Dict[str, str]) -> Dict[str, Any]:
        """
        Restores the original PII in the payload from the pii_map.
        """
        restored_payload = payload.copy()
        
        for choice in restored_payload.get("choices", []):
            if "message" in choice and "content" in choice["message"]:
                content = choice["message"]["content"]
                if content:
                    for placeholder, original_value in pii_map.items():
                        content = content.replace(placeholder, original_value)
                    choice["message"]["content"] = content
        
        return restored_payload