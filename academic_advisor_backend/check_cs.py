import json

with open("majors_data/majors_embeddings.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Total majors: {len(data)}")

# Find CS
cs = [m for m in data if "מדעי המחשב" in m.get('original_name', '')]
print(f"Computer Science found: {len(cs)}")

if cs:
    print(f"CS name: {cs[0]['original_name']}")
    print(f"CS vector length: {len(cs[0]['vector'])}")
    print(f"CS scores keys: {list(cs[0]['scores'].keys())[:5]}")
else:
    print("COMPUTER SCIENCE NOT FOUND!")

# List all majors
print("\nAll majors:")
for i, m in enumerate(data, 1):
    print(f"{i}. {m['original_name'][:50]}")
