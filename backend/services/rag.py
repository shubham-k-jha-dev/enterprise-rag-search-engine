import os
from dotenv import load_dotenv
from groq import Groq
from backend.services.search import search
from backend.services.cache import get_cached, set_cache

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def stream_answer(answer):
    words = answer.split()
    for word in words:
        yield word + " "
        
def generate_answer(query: str):
    # Check cache first
    cached = get_cached(query)

    if cached:
        print("CACHE HIT")
        return cached
    
    # Retrieve docs
    docs = search(query)
    context = "\n".join(doc["text"] for doc in docs if isinstance(doc.get("text"), str))
    
    prompt = f"""
                You are an AI assistant answering questions using the provided context.

                Context:
                {context}

                Question:
                {query}

                Instructions:
                - Use only the provided context.
                - If the answer is not in the context, say "I don't know".
                - Answer in full detail using all relevant information from the context.

                Answer:
            """
                
    # Call LLM
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    
    answer = completion.choices[0].message.content
    
    result = {
        "answer": answer,
        "docs": docs
    }
    
    # Store in cache
    set_cache(query, result)
    
    return result