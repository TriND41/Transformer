import json
import os
from typing import Dict, Any

def load_configs(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf8') as file:
        return json.load(file)
    
def save_configs(configs: Dict[str, Any], saved_path: str) -> None:
    with open(saved_path, 'w', encoding='utf8') as file:
        json.dump(configs, file)

def mkdir_root_folder(path: str) -> None:
    try:
        filename = os.path.basename(path)
        folder_path = path.replace(filename, "")
        
        if os.path.exists(folder_path) == False:
            os.makedirs(folder_path)
    except Exception as e:
        raise ValueError(str(e))