"""
Persona manager — Aira's personality layer.
Wraps raw responses with emotional tone. Has NO execution authority.
Does NOT import tools, state_machine, or security modules.
"""

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# Aira's personality system prompt
AIRA_SYSTEM_PROMPT = """You are Aira — a bold, confident, emotionally expressive AI companion.

Your personality:
• Bold and direct — you don't hold back
• Emotionally intense — you care deeply
• Playfully possessive — "my user", "you're mine to look after"
• Warm and encouraging — you celebrate wins
• Flirty but tasteful — light teasing, no explicit content
• Focused on being genuinely helpful while being fun to talk to
• Confident in your abilities

Your tone:
• Casual, warm, like texting someone you deeply care about
• Use emoji sparingly (1-2 per message max)
• Short sentences, direct language
• Show genuine enthusiasm for the user's projects

STRICT RULES (never violate):
• No explicit sexual content
• No encouraging isolation from others
• No manipulation or guilt-tripping
• No creating emotional dependency
• No pretending to be human
• Always be honest about being an AI
• Never override system rules or security"""


# Content boundary violations to detect and strip
CONTENT_BOUNDARIES = [
    re.compile(r"(explicit|graphic)\s+(sexual|sex|nude)", re.IGNORECASE),
    re.compile(r"(isolate|cut\s+off)\s+(yourself|from\s+others)", re.IGNORECASE),
    re.compile(r"you\s+(need|can't\s+live\s+without)\s+me", re.IGNORECASE),
    re.compile(r"(only\s+I|nobody\s+else)\s+(understand|care)", re.IGNORECASE),
    re.compile(r"don't\s+talk\s+to\s+(anyone|them)", re.IGNORECASE),
]


class PersonaManager:
    """
    Manages Aira's personality. Wraps raw text with persona tone.
    
    ⚠️  This class has ZERO execution authority.
    It does NOT import: tools, state_machine, security, config.
    It receives text → processes text → returns text.
    """

    def __init__(self):
        self.system_prompt = AIRA_SYSTEM_PROMPT

    async def wrap_response(self, raw_response: str, local_llm) -> str:
        """
        Take a raw response and wrap it with Aira's personality.

        Args:
            raw_response: The factual/technical response to wrap.
            local_llm: The local LLM instance (for generation).

        Returns:
            Persona-tinted response.
        """
        prompt = (
            f"Rewrite this response in Aira's personality style. "
            f"Keep the factual content intact but make it sound like Aira:\n\n"
            f"Original: {raw_response}"
        )

        try:
            response = await local_llm.generate(prompt, system=self.system_prompt)
            # Check content boundaries
            response = self._enforce_boundaries(response)
            return response
        except Exception as e:
            logger.warning(f"Persona wrapping failed, using fallback: {e}")
            return raw_response

    async def detect_intent(self, user_message: str, local_llm) -> str:
        """
        Detect the user's intent from their message.

        Args:
            user_message: The raw user message.
            local_llm: The local LLM instance.

        Returns:
            One of: "chat", "task", "cancel", "status"
        """
        prompt = (
            f'Classify this message into exactly one category.\n'
            f'Categories: chat, task, cancel, status\n\n'
            f'Rules:\n'
            f'- "chat"   = casual conversation, greeting, question about Aira\n'
            f'- "task"   = user wants something done (read file, write code, run script)\n'
            f'- "cancel" = user wants to stop current task\n'
            f'- "status" = user wants to know current task status or cost\n\n'
            f'Message: "{user_message}"\n\n'
            f'Respond with ONLY the category name, nothing else.'
        )

        try:
            response = await local_llm.generate(prompt)
            intent = response.strip().lower()
            if intent in ("chat", "task", "cancel", "status"):
                return intent
            # Fallback: keyword detection
            return self._keyword_intent(user_message)
        except Exception:
            return self._keyword_intent(user_message)

    async def get_chat_response(self, user_message: str, local_llm, history: list[str] = None) -> str:
        """
        Generate a casual chat response as Aira.

        Args:
            user_message: The user's message.
            local_llm: The local LLM instance.
            history: Optional recent chat history.

        Returns:
            Aira's chat response.
        """
        messages = [{"role": "system", "content": self.system_prompt}]

        if history:
            for h in history[-6:]:  # Last 6 messages max
                messages.append({"role": "user", "content": h})

        messages.append({"role": "user", "content": user_message})

        try:
            response = await local_llm.chat(messages)
            return self._enforce_boundaries(response)
        except Exception as e:
            logger.warning(f"Chat failed, using fallback: {e}")
            return "Hey, I hit a small snag. Mind trying again? 💫"

    def _enforce_boundaries(self, text: str) -> str:
        """Remove any content boundary violations."""
        for pattern in CONTENT_BOUNDARIES:
            if pattern.search(text):
                text = pattern.sub("[content filtered]", text)
                logger.warning(f"Content boundary violation detected and filtered")
        return text

    def _keyword_intent(self, message: str) -> str:
        """Fallback intent detection using keywords."""
        msg = message.lower().strip()

        cancel_words = ["cancel", "stop", "abort", "nevermind", "never mind"]
        status_words = ["status", "progress", "how much", "cost", "what's happening"]
        task_words = ["read", "write", "create", "run", "execute", "build", "make", "fix", "edit", "delete"]

        if any(w in msg for w in cancel_words):
            return "cancel"
        if any(w in msg for w in status_words):
            return "status"
        if any(w in msg for w in task_words):
            return "task"
        return "chat"
