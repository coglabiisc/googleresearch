import json

def import_config():
    with open('./utils/config_utils.json', 'r') as f:
        return json.load(f)