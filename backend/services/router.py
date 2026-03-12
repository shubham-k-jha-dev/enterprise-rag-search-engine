def detect_intent(query: str):
    
    q = query.lower()
    
    keyword_patterns = [
        "python",
        "fastapi",
        "framework",
        "machine learning"
    ]
    
    conversational_patterns = [
        "hello",
        "hi",
        "how are you"
    ]
    
    if any(word in q for word in conversational_patterns):
        return "chat"
    
    if any(word in q for word in keyword_patterns):
        return "keyword"
    
    return "semantic"