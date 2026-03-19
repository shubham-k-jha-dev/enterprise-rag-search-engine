import logging
import os

from dotenv import load_dotenv
from groq import APIConnectionError, APIStatusError, Groq

load_dotenv()

logger = logging.getLogger(__name__)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

MODEL_NAME: str = "llama-3.3-70b-versatile"
TEMPERATURE: float = 0.0

def expand_query(query: str) -> str:
    """
    Expands a query with related keywords, synonyms, and concepts.

    Uses the Groq LLM to generate an enriched version of the query that
    improves retrieval recall.

    NOTE: This function is called in PARALLEL with rewrite_query() inside
    search.py using ThreadPoolExecutor. 
    """
    logger.info("Expanding query: '%s'", query)

    prompt: str = (
        "Expand the following search query with related keywords, synonyms, "
        "and relevant concepts to improve document retrieval.\n"
        "Return ONLY the expanded query as a single line.\n"
        "No explanation. No bullet points. No preamble. No quotes.\n\n"
        f"Query: {query}\n"
        "Expanded query:"
    )
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=TEMPERATURE,
            max_tokens=150,
        )

        expanded: str = completion.choices[0].message.content.strip()

        logger.info("Expanded query: '%s'", expanded)
        return expanded

    except APIConnectionError:
        logger.warning(
            "Query expander: Groq connection failed — using original query: '%s'",
            query,
        )
        return query

    except APIStatusError as e:
        logger.warning(
            "Query expander: Groq API error %d — using original query: '%s'",
            e.status_code,
            query,
        )
        return query

    except Exception as e:
        logger.error(
            "Query expander: Unexpected error '%s' — using original query: '%s'",
            e,
            query,
        )
        return query