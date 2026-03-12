import json
import time
from datetime import datetime

LOG_FILE = "data/query_logs.json"

def log_query(query, docs, latency):
    log_entry = {
        "timestamp" : datetime.utcnow().isoformat(),
        "query" : query,
        "docs" : docs,
        "latency" : latency
    }
    
    try:
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        print("Logging failed: ", e)