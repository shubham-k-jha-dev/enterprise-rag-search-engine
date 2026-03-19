from backend.services.logger import get_conversation_history, save_conversation_turn


def load_history(conversation_id: str, workspace: str = "default", limit: int = 6) -> list:
    """
    Load previous turns of a conversation from SQLite.
    Returns a list of message dicts in OpenAI/Groq format:
    [
        {"role": "user", "content": "Who killed Ned Stark?"},
        {"role": "assistant", "content": "Ned Stark was executed by Joffrey..."},
        ...
    ]
    This format is what Groq expects in its messages array.
    """

    # get_conversation_history returns list of {"user": ..., "assistant": ...} dicts
    # we need to convert each turn into two separate messages
    # one with role="user" and one with role="assistant"
    # because that's the format LLMs use for multi-turn conversation
    turns = get_conversation_history(conversation_id, limit=limit)
    # limit=6 means we load at most 6 previous turns
    # loading too many turns makes the prompt very long
    # which slows down the LLM and can exceed token limits

    messages = []
    for turn in turns:
        # each turn becomes two messages — user then assistant
        messages.append({"role": "user", "content": turn["user"]})
        messages.append({"role": "assistant", "content": turn["assistant"]})

    return messages
    # example output:
    # [
    #   {"role": "user", "content": "Who killed Ned Stark?"},
    #   {"role": "assistant", "content": "Ned Stark was executed by Joffrey Baratheon..."},
    #   {"role": "user", "content": "Why did he do it?"},
    #   {"role": "assistant", "content": "Joffrey ordered it to consolidate his rule..."},
    # ]


def store_turn(conversation_id: str, workspace: str, user_message: str, assistant_message: str):
    """
    Save one completed turn (user question + assistant answer) to SQLite.
    Called after the LLM finishes generating its answer.
    """

    # delegate to logger.py which handles the actual SQLite write
    save_conversation_turn(
        conversation_id=conversation_id,
        workspace=workspace,
        user_message=user_message,
        assistant_message=assistant_message
    )
    # this persists the turn so future requests with the same
    # conversation_id can load it as context