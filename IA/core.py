"""Interface do núcleo seguro do assistente local."""

from oi import (
    DEFAULT_MODEL,
    VISION_MODEL,
    append_conversation,
    ask_local_ai,
    safe_calculate,
    try_calculate_message,
    try_local_tool,
)

__all__ = [
    "DEFAULT_MODEL",
    "VISION_MODEL",
    "append_conversation",
    "ask_local_ai",
    "safe_calculate",
    "try_calculate_message",
    "try_local_tool",
]