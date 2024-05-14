import json

from typing import Any


def get_pretty(obj: dict | list[Any]) -> str:
    return json.dumps(obj, indent=4, ensure_ascii=False)
