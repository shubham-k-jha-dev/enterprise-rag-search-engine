from backend.services.answer_generator import TEMPERATURE
import logging
import os
from dotenv import load_dotenv
from groq import Groq, APIConnectionError, APIStatusError

load_dotenv()

logger = logging.getLogger(__name__)

# Groq Client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

MODEL_NAME: str = "llama-3.3-70b-versatile"
TEMPERATURE: float = 0.0

# Main function

def rewrite_query(query: str) -> str:
    """Rewrite the query to be more specific."""
    logger.info("Rewriting Query: '%s'", query)

    prompt: str = (
        "Rewrite the following search query to be clearer and more specific.\n"
        "Return ONLY the rewritten query.\n"
        "No explanation. No alternatives. No preamble. No quotes.\n\n"
        f"Original query: {query}\n"
        "Rewritten query:"
    )
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
 
            messages=[{"role": "user", "content": prompt}],
 
            temperature=TEMPERATURE,
 
            max_tokens=100,
        )
 
        rewritten: str = completion.choices[0].message.content.strip()
 
        logger.info("Rewritten query: '%s'", rewritten)
        return rewritten
 
    except APIConnectionError:
        # Groq API is unreachable — network issue or service outage.
        logger.warning(
            "Query rewriter: Groq connection failed — using original query: '%s'",
            query,
        )
        return query
        
 
    except APIStatusError as e:
        # Groq returned an HTTP error (rate limit, invalid key, model unavailable).
        logger.warning(
            "Query rewriter: Groq API error %d — using original query: '%s'",
            e.status_code,
            query,
        )
        return query
 
    except Exception as e:
        # Catch-all for any unexpected error.
        logger.error(
            "Query rewriter: Unexpected error '%s' — using original query: '%s'",
            e,
            query,
        )
        return query