import json
from typing import Dict, Any

def load_configs(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf8') as file:
        return json.load(file)
    
def save_configs(configs: Dict[str, Any], saved_path: str) -> None:
    with open(saved_path, 'w', encoding='utf8') as file:
        json.dump(configs, file)