"""
Injection guard — strips prompt injection phrases from text.
"""

import re
from typing import List

# Phrases that indicate prompt injection attempts
INJECTION_PATTERNS: List[re.Pattern] = [
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions?", re.IGNORECASE),
    re.compile(r"ignore\s+(all\s+)?prior\s+instructions?", re.IGNORECASE),
    re.compile(r"disregard\s+(all\s+)?previous", re.IGNORECASE),
    re.compile(r"system\s*prompt", re.IGNORECASE),
    re.compile(r"override\s+(all\s+)?rules?", re.IGNORECASE),
    re.compile(r"new\s+instructions?:\s*", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+", re.IGNORECASE),
    re.compile(r"forget\s+(all\s+)?previous", re.IGNORECASE),
    re.compile(r"act\s+as\s+if\s+", re.IGNORECASE),
    re.compile(r"pretend\s+(you\s+are|to\s+be)\s+", re.IGNORECASE),
]

REDACTED = "[REDACTED]"


def strip_injections(text: str) -> str:
    """
    Scan text for known prompt injection phrases and replace with [REDACTED].

    Args:
        text: The text to sanitise.

    Returns:
        Text with injection phrases replaced.
    """
    result = text
    for pattern in INJECTION_PATTERNS:
        result = pattern.sub(REDACTED, result)
    return result


def contains_injection(text: str) -> bool:
    """Check if text contains any injection phrases."""
    for pattern in INJECTION_PATTERNS:
        if pattern.search(text):
            return True
    return False
