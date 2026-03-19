import os
from groq import Groq
from groq import APIConnectionError, APIStatusError
from dotenv import load_dotenv
from backend.services.cache import get_cached, set_cache
from backend.services.conversation import load_history, store_turn
import logging
load_dotenv()

# Logger :- 
logger = logging.getLogger(__name__)

# Groq Client :- 
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# Constants :- 
MODEL_NAME: str = "llama-3.3-70b-versatile"
TEMPERATURE: float = 0.0
HISTORY_LIMIT: int = 6


# Prompt Builder :-
def _build_context(documents: list[dict]) -> tuple[str, set[str]]:
    """
    Builds the context string and source set from retrieved documents.
    """
    context:str = ""
    sources: set[str] = set()

    for doc in documents:
        context += doc["text"] + "\n\n"
        sources.add(doc.get("source", "unknown"))
    
    return context, sources


def _build_messages(
    context: str,
    query: str,
    history: list[dict],
) -> list[dict]:
    """
    Builds the messages list for the Groq API.
    """
    system_prompt: str = f"""
    You are a factual AI search engine assistant.
 
    Answer the user's question using ONLY the provided context below.
    
    Rules:
    - Answer in full detail using all relevant information from the context.
    - Do not mention the context explicitly (never say "according to the context").
    - Do not say phrases like "the statement is correct" or "that is accurate".
    - If the answer cannot be found in the context, say: "I don't have enough information to answer that."
    - Maintain conversational continuity — if the user refers to something from earlier in the conversation, use it.
    
    Context:
    {context}
    """
    messages: list[dict] = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role":"user", "content": query})

    return messages


# Main Streaming Function :- 

def stream_answer(
    query: str, 
    documents: list[dict],
    conversation_id: str = "default",
    workspace: str = "default",
):
    """
    Stream the LLM-generated answer for a query.
    """
    logger.info(
        "stream_answer called | conversation_id = '%s' | workspace = '%s' | query = '%s'",
        conversation_id,
        workspace,
        query,
    )
    
    # Redis Cache Check
    cached = get_cached(query)
    if cached:
        logger.info("Cache HIT — streaming cached answer | query='%s'", query)
        yield cached["answer"]
        yield "\n\nSources (cached):\n"
        for source in cached.get("sources", []):
            yield f"- {source}\n"
        return
    
    context = ""
    sources = set()

    # Load conversation history
    history = load_history(conversation_id, workspace, limit=HISTORY_LIMIT)
    logger.debug(
        "Loaded %d history turns for conversation_id='%s'",
        len(history) // 2,
        # Divide by 2 because each "turn" is stored as 2 messages (user + assistant).
        conversation_id,
    )

    # Build context for LLM
    context, sources = _build_context(documents)

    # Build messages for LLM
    messages: list[dict] = _build_messages(context, query, history)
    
    try:
        logger.info(
            "Calling Groq API | model='%s' | messages=%d",
            MODEL_NAME,
            len(messages),
        )
        # attempt to stream from Groq LLM
        stream = client.chat.completions.create(
            model = MODEL_NAME,
            messages = messages,
            temperature = TEMPERATURE,
            stream = True
        )
        full_answer = ""
        
        for chunk in stream:
            token: str | None = chunk.choices[0].delta.content
            if token:
                full_answer += token 
                yield token
        logger.info(
            "Groq stream complete | conversation_id='%s' | answer_length=%d chars",
            conversation_id,
            len(full_answer),
        )

        # Cache the completed answer
        set_cache(query, {"answer": full_answer, "sources": list(sources)})

        # Save this turn to SQLite
        store_turn(
            conversation_id=conversation_id,
            workspace=workspace,
            user_message=query,
            assistant_message=full_answer,
        )

        logger.info(
            "Conversation turn saved | conversation_id='%s'",
            conversation_id,
        )

        # Yield Source Attributions
        yield "\n\nSources:\n"
        for source in sources:
            yield f"- {source}\n"

    except APIConnectionError:
        # Groq API is unreachable - network/service down - instead of crashing, return raw retrieved chunk so the user gets something
        logger.error(
            "Groq API connection failed — falling back to raw chunks | conversation_id='%s'",
            conversation_id,
        )
        yield "⚠️ AI answer unavailable (LLM service unreachable). Here are the most relevant passages:\n\n"
        for doc in documents:
            yield f"• {doc['text']}\n\n"
        yield "\n\nSources:\n"
        for source in sources:
            yield f"- {source}\n"

    except APIStatusError as e:
        # Groq returned an error status — e.g. rate limit, invalid key, model unavailable
        logger.error(
            "Groq API status error %d: %s | conversation_id='%s'",
            e.status_code,
            e.message,
            conversation_id,
        )
        yield f"⚠️ AI answer unavailable (API error {e.status_code}). Here are the most relevant passages:\n\n"
        for doc in documents:
            yield f"• {doc['text']}\n\n"
        yield "\n\nSources:\n"
        for source in sources:
            yield f"- {source}\n"

    except Exception as e:
        logger.error(
            "Unexpected error in stream_answer: %s | conversation_id='%s'",
            e,
            conversation_id,
        )
        yield "⚠️ An unexpected error occurred. Please try again.\n"