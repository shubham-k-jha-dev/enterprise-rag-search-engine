import json

LOG_FILE = "data/query_logs.json"

queries = []

with open(LOG_FILE, "r") as f:
    content = f.read()

decoder = json.JSONDecoder()
idx = 0

while idx < len(content):
    content = content[idx:].lstrip()
    if not content:
        break
    obj, end = decoder.raw_decode(content)
    queries.append(obj)
    idx = end

print("Total Queries Logged:", len(queries))

sources = {}

for q in queries:
    for doc in q["docs"]:
        if isinstance(doc, dict):
            src = doc.get("source", "unknown")
        else:
            src = "unknown"
        sources[src] = sources.get(src, 0) + 1

print("\nDocument Source Distribution:\n")

for k, v in sources.items():
    print(k, ":", v)