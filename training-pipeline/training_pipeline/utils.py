import yaml
import json


def load_yaml(path) -> dict:

    with path.open("r") as f:
        config = yaml.safe_load(f)

    return config


def load_json(path) -> dict:

    with path.open("r") as f:
        data = json.load(f)

    return data


def write_json(data, path):
    with path.open("w") as f:
        json.dump(data, f, indent=4)
