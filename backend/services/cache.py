import redis
import json
import os
from backend.services.metrics import record_cache_hit

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
redis_client = redis.Redis(
    host = REDIS_HOST,
    port = 6379,
    decode_responses = True
)

def get_cached(query):
    data = redis_client.get(query)
    
    if data:
        record_cache_hit()
        return json.loads(data)
    
    return None

def set_cache(query, result):
    redis_client.set(query, json.dumps(result), ex = 3600)