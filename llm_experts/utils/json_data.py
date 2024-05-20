import json

from typing import Any


def get_pretty(obj: dict | list[Any]) -> str:
    return json.dumps(obj, indent=4, ensure_ascii=False)


def load_json(json_file_path: str) -> dict | list[dict]:
    with open(json_file_path, "r") as f:
        content = json.loads(f.read())
        return content
