import os
from groq import Groq
from dotenv import load_dotenv
from backend.services.cache import get_cached, set_cache

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def stream_answer(query, documents):
    
    # check Redis first — if the exact same query was answered before, we stream it from cache without hitting the LLM
    cached = get_cached(query)

    if cached:
        print("CACHE HIT — returning cached answer")
        yield cached["answer"]
        yield "\n\nSources (cached):\n"
        for source in cached.get("sources", []):
            yield f"- {source}\n"
        return
    
    context = ""
    sources = set()

    # build context for LLM
    for doc in documents:
        context += doc["text"] + "\n\n"
        sources.add(doc.get("source", "unknown"))
    prompt = f"""
You are a factual AI search engine.

Answer the user question using ONLY the provided context.

Rules:
- Answer in full detail using all relevant information from the context.
- Do not mention the context explicitly.
- Do not say phrases like "the statement is correct" or "that is accurate".

Context:
{context}

User Question:
{query}

Answer:
"""

    stream = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        stream=True
    )

    
    full_answer = ""
    
    for chunk in stream:
        token = chunk.choices[0].delta.content
        if token:
            full_answer += token 
            yield token
            
    # Cache the full answer after streaming completes (TTL = 1 hour)
    set_cache(query, {"answer": full_answer, "sources": list(sources)})

    # after streaming answer, append sources
    yield "\n\nSources:\n"

    for source in sources:
        yield f"- {source}\n"