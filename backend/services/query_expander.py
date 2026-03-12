import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key = os.getenv("GROQ_API_KEY"))


def expand_query(query: str):
    prompt = f"""
                Expand the following search query with related keywords and concepts.
                Return only the expanded query.
                
                Query : {query}
                Expanded query :
                """
    
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    
    expanded_query = completion.choices[0].message.content.strip()
    
    return expanded_query