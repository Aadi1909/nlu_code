import json

input_path = "training_data.jsonl"
output_path = "structure_training_data.json"

structured_data = []

with open(input_path, "r", encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)

        item = {
            "text": row.get("text", ""),
            "intent": row.get("intent", "unknown_intent"),
            "entities": row.get("entities", []),
            "language": row.get("language", "en"),
            "metadata": {
                "scenario": row.get("scenario", "general"),
                "complexity": row.get("complexity", "simple")
            }
        }

        structured_data.append(item)

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(structured_data, f, indent=4, ensure_ascii=False)

print("✅ Converted file saved at:", output_path)
