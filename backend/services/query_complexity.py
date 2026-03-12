def query_complexity(query: str):
    query = query.lower()
    # comparison queries
    if "compare" in query or "difference" in query or "vs" in query:
        return "complex"
    
    # analytical queries
    if "explain" in query or "why" in query or "how" in query:
        return "complex"
    
    # long queries
    if len(query.split()) > 8:
        return "complex"
    
    return "simple"