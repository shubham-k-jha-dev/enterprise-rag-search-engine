from backend.services.search import search
from backend.services.query_planner import plan_query
from backend.services.search import search


def plan_subqueries(query: str):
    """
    Generate distinct subqueries for agentic retrieval.
    """
    
    query_clean = query.strip()
    
    subqueries = [query_clean]
    
    lower_query = query_clean.lower()
    
    if " and " in lower_query:
        parts = query_clean.split(" and")
        subqueries.extend(parts)
        
    elif "why" in lower_query:
        reason_query = query_clean.replace("Why", "Reason for")
        reason_query = query_clean.replace("why", "reason for")
        
        if reason_query != query_clean:
            subqueries.append(reason_query)
            
    # Remove duplicates
    unique_subqueries = list(dict.fromkeys(subqueries))
    
    return unique_subqueries
        


def execute_subquery_retrieval(subqueries, workspace):
    """
    Execute retrieval for each planned subquery.
    """
    all_results = []
    
    for q in subqueries:
        print(f"AGENT SEARCHING: {q}")
        
        results = search(query = q, workspace=workspace)
        all_results.extend(results)
        
    return all_results


def deduplicate_results(results):
    """
    Remove duplicate document chunks retrieved by multiple subqueries.
    """
    
    seen_ids = set()
    unique_results = []
    
    for r in results:
        source = r.get("source", "unknown")
        chunk_id = r.get("chunk_id", r.get("text", ""))
        
        key = (source, chunk_id)
        
        if key not in seen_ids:
            seen_ids.add(key)
            unique_results.append(r)
            
    print(f"DEDUPLICATED RESULTS: {len(unique_results) } unique_chunks")
    return unique_results

def run_agentic_retrieval(query: str, workspace: str):
    """
    Full agentic retrieval pipeline.
    """

    print(f"\nAGENTIC RETRIEVAL STARTED")
    print(f"ORIGINAL QUERY: {query}")

    subqueries = plan_subqueries(query)

    print(f"PLANNED SUBQUERIES: {subqueries}")

    results = execute_subquery_retrieval(subqueries, workspace)

    unique_results = deduplicate_results(results)

    return unique_results