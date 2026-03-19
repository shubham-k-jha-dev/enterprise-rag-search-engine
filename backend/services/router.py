import logging

# Logger 
logger = logging.getLogger(__name__)

CONVERSATIONAL_PATTERNS: tuple[str, ...] = (
    "hello",
    "hi",
    "hey",
    "good morning",
    "good evening",
    "how are you",
    "what's up",
    "thanks",
    "thank you",
    "bye",
    "goodbye",
)

KEYWORD_PATTERNS: tuple[str, ...] = (
    # ── Game of Thrones — character names ────────────────────────────────────
    "ned stark",
    "jon snow",
    "daenerys",
    "cersei",
    "tyrion",
    "joffrey",
    "arya",
    "sansa",
    "robb stark",
    "bran",
    "jaime",
    "lannister",
    "targaryen",
    "baratheon",
    "stark",
 
    # ── Game of Thrones — locations and objects ───────────────────────────────
    "iron throne",
    "winterfell",
    "king's landing",
    "the wall",
    "castle black",
    "night's watch",
    "dragon",
    "westeros",
    "seven kingdoms",
    "red wedding",
    "night king",
 
    # ── Dexter — character names ──────────────────────────────────────────────
    "dexter",
    "dexter morgan",
    "debra",
    "debra morgan",
    "harry morgan",
    "trinity killer",
    "arthur mitchell",
    "doakes",
    "sergeant doakes",
    "miami metro",
 
    # ── Dexter — concepts and locations ──────────────────────────────────────
    "dark passenger",
    "code of harry",
    "blood spatter",
    "kill room",
    "miami",
)


# Main routing function
def detect_intent(query: str) -> str:
    """
    Classifier the query into one of three intent categories :- 
    1. chat
    2. keyword
    3. semantic
    """
    query_lower: str = query.lower().strip()
    if any(pattern in query_lower for pattern in CONVERSATIONAL_PATTERNS):
        logger.info("Intent detected: 'chat' | query = '%s'", query)
        return "chat"
    
    if any(pattern in query_lower for pattern in KEYWORD_PATTERNS):
        logger.info("Intent detected: 'keyword' | query='%s'", query)
        return "keyword"
    
    logger.info("Intent detected: 'semantic' | query='%s'", query)
    return "semantic"