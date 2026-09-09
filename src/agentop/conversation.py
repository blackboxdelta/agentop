"""Prompt construction for solo and multi-model Ollama conversations."""
from __future__ import annotations

from dataclasses import dataclass


MAX_DISCUSSION_CHARS = 24_000


@dataclass(frozen=True)
class ConversationEntry:
    speaker: str
    content: str


def build_solo_messages(
    entries: list[ConversationEntry],
    *,
    model: str,
) -> list[dict[str, str]]:
    messages = [
        {
            "role": "system",
            "content": (
                f"You are {model}, running locally through AgentOp. "
                "Answer the user directly and concisely."
            ),
        }
    ]
    messages.extend(
        {
            "role": "user" if entry.speaker == "User" else "assistant",
            "content": entry.content,
        }
        for entry in entries
    )
    return messages


def build_roundtable_messages(
    entries: list[ConversationEntry],
    *,
    topic: str,
    model: str,
    participants: tuple[str, ...],
    round_number: int,
) -> list[dict[str, str]]:
    peers = ", ".join(participant for participant in participants if participant != model)
    discussion = "\n\n".join(
        f"{entry.speaker}:\n{entry.content}" for entry in entries
    )
    if len(discussion) > MAX_DISCUSSION_CHARS:
        discussion = "[Earlier discussion omitted]\n\n" + discussion[-MAX_DISCUSSION_CHARS:]
    return [
        {
            "role": "system",
            "content": (
                f"You are {model}, participating in a local model roundtable with {peers}. "
                "Respond to the discussion as yourself, consider and challenge earlier "
                "statements when useful, and do not impersonate another participant. "
                "Keep your response under 180 words so the conversation progresses."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Topic: {topic}\n"
                f"Round: {round_number}\n\n"
                f"Discussion so far:\n{discussion}\n\n"
                f"Continue the discussion as {model}."
            ),
        },
    ]
