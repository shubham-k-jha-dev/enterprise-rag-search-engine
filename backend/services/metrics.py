metrics = {
    "total_queries" : 0,
    "cache_hits": 0,
    "total_latency" : 0
}

def record_query(latency):
    metrics["total_queries"] += 1
    metrics["total_latency"] += latency

# When Redis returns a cached result
def record_cache_hit():
    metrics["cache_hits"] += 1
    
def get_metrics():
    
    total = metrics["total_queries"]
    
    avg_latency = 0
    cache_hit_rate = 0
    
    if total > 0:
        avg_latency = metrics["total_latency"] / total
        cache_hit_rate = metrics["cache_hits"] / total
        
    return {
        "total_queries" : total,
        "avg_latency_ms" : round(avg_latency, 2),
        "cache_hit_rate" : round(cache_hit_rate, 3)
    }
    