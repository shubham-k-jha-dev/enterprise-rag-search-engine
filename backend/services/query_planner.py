import logging
from backend.services.logger import get_workspace_names


# Logger :- 
logger = logging.getLogger(__name__)

# Constants :- 
COMPARISON_TRIGGERS: tuple[str, ...] = (
    "compare",
    "comparison",
    "difference",
    "differences",
    "vs",
    "versus",
    "contrast",
    "similar",
    "similarity",
    "both",
)

EXCLUDED_FROM_COMPARISON: set[str] = {"default"}

# Main function

def plan_query(query: str, workspace: str) -> dict:
    """
    Analyses the query and returns a routing plan.
 
    The plan tells the agentic retrieval system:
        - What type of query this is (single vs comparison)
    """
    query_lower: str = query.lower().strip()
    is_comparison = any(trigger in query_lower for trigger in COMPARISON_TRIGGERS)
    if is_comparison:
        all_names: list[str] = get_workspace_names()
        comparison_workspaces: list[str] = [
            name for name in all_names
            if name not in EXCLUDED_FROM_COMPARISON
        ]

        if not comparison_workspaces:
            logger.warning(
                "No workspaces available for comparison — falling back to single"
            )
            comparison_workspaces = [workspace]
        plan: dict = {
                "type": "comparison",
                "workspaces": comparison_workspaces,
        }
        logger.info(
                "Query plan: COMPARISON | workspaces=%s | query='%s'",
                comparison_workspaces,
                query,
        )
    else:
        plan: dict = {
            "type": "single",
            "workspaces": [workspace],
            # Only the user's selected workspace is searched.
        }
        logger.info(
            "Query plan: SINGLE | workspace='%s' | query='%s'",
            workspace,
            query,
        )
 
    return plan