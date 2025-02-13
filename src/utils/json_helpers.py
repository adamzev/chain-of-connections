import json


def load_json(file_path):
    is_jsonl = file_path.endswith('.jsonl')
    with open(file_path, 'r') as file:
        if is_jsonl:
            return [json.loads(line) for line in file]
        else:
            return json.load(file)


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (set, frozenset)):
            return list(obj)  # Convert sets to lists for JSON serialization
        return super().default(obj)


def dump_json(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4, cls=CustomJSONEncoder)
