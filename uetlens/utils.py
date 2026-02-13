import os
import json
from typing import List, Dict, Any, Optional
import torch
import transformers


def load_text_data(file_path: str) -> List[str]:

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Text file not found: {file_path}")
        
    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    
    if not lines:
        raise ValueError(f"No valid lines found in {file_path}")
        
    return lines


def save_json_results(results: Dict[str, Any], output_path: str) -> None:

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def setup_tokenizer(model_path: str) -> transformers.AutoTokenizer:

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    return tokenizer


def get_available_device() -> str:

    if torch.cuda.is_available():
        return f"cuda:{torch.cuda.current_device()}"
    else:
        return "cpu"


def load_feature_metadata() -> Dict[str, Any]:

    return {}


def validate_layer_indices(layers: List[int], max_layer: int = 32) -> List[int]:

    valid_layers = [l for l in layers if 0 <= l < max_layer]
    if not valid_layers:
        raise ValueError(f"No valid layer indices found in {layers}")
    return valid_layers


class ProgressLogger:

    def __init__(self, total_items: int, description: str = "Processing"):
        self.total_items = total_items
        self.description = description
        self.current = 0
    
    def update(self, increment: int = 1) -> None:
        self.current += increment
        if self.current % 10 == 0 or self.current == self.total_items:
            print(f"{self.description}: {self.current}/{self.total_items}")
    
    def finish(self) -> None:
        print(f"{self.description}: Complete!")


def format_feature_name(feature_path: str) -> str:
    filename = os.path.basename(feature_path)
    name = os.path.splitext(filename)[0]
    return name.replace("_", " ").replace("-", " ").title()