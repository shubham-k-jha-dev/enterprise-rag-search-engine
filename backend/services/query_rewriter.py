import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key = os.getenv("GROQ_API_KEY"))

def rewrite_query(query: str):
    prompt = f"""Rewrite the following search query to be clearer and more specific.
                Return ONLY the rewritten query. No explanation. No alternatives. No preamble.

                Original query: {query}
                Rewritten query:
                """
    
    completion = client.chat.completions.create(
        model = "llama-3.3-70b-versatile",
        messages = [{"role": "user", "content" : prompt}],
        temperature = 0
    )
    
    improved_query = completion.choices[0].message.content.strip()
    
    return improved_query