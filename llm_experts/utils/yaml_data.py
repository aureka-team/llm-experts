import yaml


def load_yaml(file_path: str) -> dict:
    with open(file_path, "r") as f:
        content = yaml.safe_load(f)
        return content


def save_yaml(dict_data: dict, file_path: str):
    with open(file_path, "w") as f:
        f.write(yaml.dump(dict_data))
