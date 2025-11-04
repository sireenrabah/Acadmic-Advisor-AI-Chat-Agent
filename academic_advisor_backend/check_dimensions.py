import json
from embeddings.embeddings import get_criteria_keys

keys = get_criteria_keys()
print(f"Expected vector dimension: {len(keys)}")
print(f"Criteria keys: {keys}")

with open("majors_data/majors_embeddings.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"\nTotal majors: {len(data)}")

# Check all vector lengths
for i, m in enumerate(data, 1):
    vec_len = len(m['vector'])
    name = m['original_name'][:40]
    match = "✓" if vec_len == len(keys) else f"✗ (got {vec_len})"
    print(f"{i:2d}. {name:40s} {match}")
