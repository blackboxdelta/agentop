from agentop.conversation import (
    MAX_DISCUSSION_CHARS,
    ConversationEntry,
    build_roundtable_messages,
    build_solo_messages,
)


def test_solo_messages_preserve_user_and_assistant_history():
    messages = build_solo_messages(
        [
            ConversationEntry("User", "first"),
            ConversationEntry("model-a", "answer"),
            ConversationEntry("User", "follow-up"),
        ],
        model="model-a",
    )

    assert [message["role"] for message in messages] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert messages[-1]["content"] == "follow-up"


def test_roundtable_prompt_names_peers_and_includes_prior_responses():
    messages = build_roundtable_messages(
        [
            ConversationEntry("User", "topic"),
            ConversationEntry("model-a", "first opinion"),
        ],
        topic="topic",
        model="model-b",
        participants=("model-a", "model-b", "model-c"),
        round_number=2,
    )

    assert "model-a, model-c" in messages[0]["content"]
    assert "model-a:\nfirst opinion" in messages[1]["content"]
    assert "Round: 2" in messages[1]["content"]


def test_roundtable_prompt_bounds_discussion_size():
    messages = build_roundtable_messages(
        [ConversationEntry("model-a", "x" * (MAX_DISCUSSION_CHARS + 100))],
        topic="topic",
        model="model-b",
        participants=("model-a", "model-b"),
        round_number=1,
    )

    assert "[Earlier discussion omitted]" in messages[1]["content"]
    assert len(messages[1]["content"]) < MAX_DISCUSSION_CHARS + 250
