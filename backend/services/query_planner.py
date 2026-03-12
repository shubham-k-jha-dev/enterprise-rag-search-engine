def plan_query(query: str, workspace: str):
    q = query.lower()
    
    if "compare" in q or "difference" in q or "vs" in q:
        return {
            "type": "comparison",
            "workspaces" : ["got", "dexter"]
        }
        
    return {
        "type": "single",
        "workspaces": [workspace]
    }