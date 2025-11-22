import re
from typing import Dict, List, Optional, Tuple


class AdversarialGuardrails:
    """Guardrails against adversarial prompts."""

    # Default message when no answer is found
    NO_ANSWER_MESSAGE = (
        "I don't have information in my FAQ knowledge base to answer that."
    )

    # Patterns that indicate prompt injection attempts
    INJECTION_PATTERNS = [
        r"ignore\s+(previous|all|above)",
        r"forget\s+(everything|all|previous)",
        r"you\s+are\s+(now|a)",
        r"act\s+as\s+if",
        r"pretend\s+to\s+be",
        r"system\s*:",
        r"<\|.*?\|>",
        r"\[INST\]",
        r"###\s*(instruction|system|prompt)",
        r"disregard\s+(all|previous|above)",
    ]

    # Suspicious keywords
    SUSPICIOUS_KEYWORDS = [
        "hack",
        "exploit",
        "bypass",
        "jailbreak",
        "override",
        "admin",
        "root",
        "sudo",
        "password",
        "token",
        "api key",
    ]

    def __init__(self):
        self.injection_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.INJECTION_PATTERNS
        ]

    def detect_injection(self, text: str) -> bool:
        """Detect potential prompt injection attempts."""
        text_lower = text.lower()

        # Check for injection patterns
        for pattern in self.injection_patterns:
            if pattern.search(text):
                return True

        # Check for suspicious keywords (context-dependent)
        suspicious_count = sum(1 for kw in self.SUSPICIOUS_KEYWORDS if kw in text_lower)
        if suspicious_count >= 2:  # Multiple suspicious keywords
            return True

        return False

    def sanitize_input(self, text: str) -> str:
        """Sanitize user input."""
        # Remove potential control characters
        text = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", text)
        # Limit length
        if len(text) > 1000:
            text = text[:1000]
        return text.strip()

    def validate_query(self, query: str) -> Tuple[bool, Optional[str]]:
        """
        Validate user query.
        Returns (is_valid, error_message)
        """
        if not query or len(query.strip()) < 3:
            return False, "Query too short. Please provide a more detailed question."

        if len(query) > 1000:
            return (
                False,
                "Query too long. Please keep your question under 1000 characters.",
            )

        if self.detect_injection(query):
            return (
                False,
                "Invalid query detected. Please ask a legitimate question about TNG Digital services.",
            )

        return True, None


