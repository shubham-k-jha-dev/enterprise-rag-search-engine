import logging
import re
from backend.services.search import search
from backend.services.query_planner import plan_query

# Logger :- 
logger = logging.getLogger(__name__)

# Constants :- 
MIN_SUBQUERY_WORDS: int = 3

# Main function

def plan_subqueries(query: str):
    """
    Generate distinct subqueries for agentic retrieval.
    """
    query_clean: str = query.strip()
    subqueries: list[str] = [query_clean]
    query_lower: str = query_clean.lower()
    
    # Case 1 : "and" queries - split into parts
    if " and " in query_lower:
        parts: list[str] = query_clean.split(" and ")
        for part in parts:
            cleaned_part: str = re.sub(r'[^\w\s]', '', part).strip()
            word_count: int = len(cleaned_part.split())
            if word_count >= MIN_SUBQUERY_WORDS:
                subqueries.append(cleaned_part)
    
    # Case 2 : "why" queries - replace with "reason for"
    elif "why" in query_lower:
        reason_query: str = query_clean.replace("why", "reason for", 1)
        
        if reason_query != query_clean:
            subqueries.append(reason_query)
            
    # Remove duplicates
    unique_subqueries: list[str] = list(dict.fromkeys(subqueries))
    
    logger.info(
        "Sub-queries planned: %d | queries=%s",
        len(unique_subqueries),
        unique_subqueries,
    )

    return unique_subqueries
        

# SUB-QUERY EXECUTION

def execute_subquery_retrieval(
    subqueries: list[str],
    workspaces: list[str],
) -> list[dict]:
    """
    Execute retrieval for each planned subquery.
    """
    all_results: list[dict] = []
    total_calls: int = len(subqueries) * len(workspaces)
    logger.info(
        "Executing %d search calls (%d sub-queries × %d workspaces)",
        total_calls,
        len(subqueries),
        len(workspaces),
    )

    for workspace in workspaces:
        for subquery in subqueries:
            logger.info(
                "Agent searching | workspace = '%s' | subquery = '%s'",
                workspace,
                subquery,
            )
            results: list[dict] = search(query = subquery, workspace = workspace)
            all_results.extend(results)
    
    logger.info(
        "All agent searches complete | total raw results = %d",
        len(all_results),
    )

    return all_results

# DEDUPLICATION

def deduplicate_results(results: list[dict]) -> list[dict]:
    """
    Remove duplicate document chunks retrieved by multiple subqueries.
    """
    
    seen_keys: set[tuple] = set()
    unique_results: list[dict] = []
    
    for result in results:
        doc_source: str = result.get("source", "unknown")
        doc_chunk_id = result.get("chunk_id", result.get("text", ""))

        key: tuple = (doc_source, doc_chunk_id)
        
        if key not in seen_keys:
            seen_keys.add(key)
            unique_results.append(result)
    logger.info(
        "Deduplication complete | before=%d | after=%d | removed=%d",
        len(results),
        len(unique_results),
        len(results) - len(unique_results),
        # Log exactly how many duplicates were removed.
    )
            
    return unique_results

def run_agentic_retrieval(query: str, workspace: str) -> list[dict]:
    """
    Full agentic retrieval pipeline.
    """
    logger.info(
        "Agentic retrieval started | workspace='%s' | query='%s'",
        workspace,
        query,
    )
    # Stage 1a: Plan which workspaces to search
    query_plan: dict = plan_query(query, workspace)
    
    workspaces_to_search: list[str] = query_plan["workspaces"]
    
    query_type: str = query_plan["type"]

    logger.info(
        "Query plan | type='%s' | workspaces=%s",
        query_type,
        workspaces_to_search,
    )

    # Stage 1b: Plan sub-queries
    subqueries: list[str] = plan_subqueries(query)
    raw_results: list[dict] = execute_subquery_retrieval(
        subqueries=subqueries,
        workspaces=workspaces_to_search,
    )
    unique_results: list[dict] = deduplicate_results(raw_results)
    logger.info(
        "Agentic retrieval complete | workspace='%s' | "
        "sub-queries=%d | workspaces=%d | final_results=%d",
        workspace,
        len(subqueries),
        len(workspaces_to_search),
        len(unique_results),
    )
 
    return unique_results