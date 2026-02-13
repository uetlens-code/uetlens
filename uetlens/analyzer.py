import os
import sys
from typing import Dict, List, Any, Optional, Union
import torch
import transformers

current_file = os.path.abspath(__file__)
uetlens_dir = os.path.dirname(os.path.dirname(current_file))
opensae_src_dir = os.path.join(uetlens_dir, "OpenSAE", "src")

if opensae_src_dir not in sys.path:
    sys.path.insert(0, opensae_src_dir)

from opensae.transformer_with_sae import TransformerWithSae
from opensae import OpenSae

from .metrics import compute_layer_stats, get_top_features_by_frc
from .utils import load_text_data, setup_tokenizer, get_available_device, ProgressLogger

class Analyzer:
    def __init__(
        self,
        model_path: str,
        sae_path_template: str = "/path/to/sae/layer_{:02d}",
        device: Optional[str] = None,
        use_multi_gpu: bool = False
    ):
        self.model_path = model_path
        self.sae_path_template = sae_path_template
        
        if use_multi_gpu and torch.cuda.device_count() > 1:
            self.device = "cuda"
            self.use_multi_gpu = True
        else:
            self.device = device or get_available_device()
            self.use_multi_gpu = False
            
        self.tokenizer = setup_tokenizer(model_path)
        self._model_cache = {}
    
    def _get_sae_model(self, layer_idx: int) -> TransformerWithSae:
        if layer_idx in self._model_cache:
            return self._model_cache[layer_idx]
        
        sae_path = self.sae_path_template.format(layer_idx)
        
        if not os.path.exists(sae_path):
            raise FileNotFoundError(f"SAE model not found: {sae_path}")
        
        sae = OpenSae.from_pretrained(sae_path)
        model = TransformerWithSae(
            self.model_path, 
            sae, 
            self.device, 
            use_multi_gpu=self.use_multi_gpu
        )
        
        self._model_cache[layer_idx] = model
        
        return model
    
    def analyze_feature(
        self,
        feature_file: str,
        layers: List[int],
        top_k: int = 10
    ) -> Dict[str, Any]:
        lines = load_text_data(feature_file)
        
        results = {
            "feature_file": feature_file,
            "total_examples": len(lines),
            "layers_analyzed": layers,
            "top_k": top_k,
            "layer_results": {},
            "unified_results": []
        }
        
        progress = ProgressLogger(len(layers), "Analyzing layers")
        
        for layer_idx in layers:
            try:
                layer_stats = self._analyze_single_layer(lines, layer_idx)
                
                if layer_stats:
                    top_features = get_top_features_by_frc(layer_stats, top_k)
                    
                    results["layer_results"][layer_idx] = {
                        "total_base_vectors": len(layer_stats),
                        "top_features": top_features,
                        "full_stats": layer_stats
                    }
                    
                    feature_name = os.path.splitext(os.path.basename(feature_file))[0]
                    for base_vector, frc_score in top_features[:3]:
                        stats = layer_stats[base_vector]
                        results["unified_results"].append({
                            "feature": feature_name,
                            "layer": layer_idx,
                            "base_vector": int(base_vector),
                            "ps": stats["ps"],
                            "pn": stats["pn"], 
                            "frc": stats["frc"],
                            "avg_max_activation": stats["avg_max_activation"]
                        })
                
                progress.update()
                
            except Exception as e:
                print(f"⚠️ Error analyzing layer {layer_idx}: {e}")
                results["layer_results"][layer_idx] = {"error": str(e)}
                progress.update()
        
        progress.finish()
        return results
    
    def _analyze_single_layer(self, lines: List[str], layer_idx: int) -> Dict[int, Dict[str, float]]:
        model = self._get_sae_model(layer_idx)
        
        encodings = self.tokenizer(
            lines, 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        ).to(self.device)
        
        structured_data = model.extract_data(encodings, self.tokenizer)
        
        return compute_layer_stats(structured_data)
    
    def extract_structured_data_batch(self, lines: List[str], layer_idx: int) -> List[Dict]:
        model = self._get_sae_model(layer_idx)
        encodings = self.tokenizer(
            lines,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)
        return model.extract_data(encodings, self.tokenizer)
    
    def analyze_structured_data(self, all_structured_data: List[Dict], top_k: int = 10) -> tuple:
        stats = compute_layer_stats(all_structured_data)
        top_features = get_top_features_by_frc(stats, top_k)
        return top_features, stats
    
    def batch_analyze_features(
        self,
        feature_files: List[str],
        layers: List[int],
        output_dir: str,
        top_k: int = 10
    ) -> Dict[str, Any]:
        os.makedirs(output_dir, exist_ok=True)
        
        batch_results = {
            "total_features": len(feature_files),
            "layers_analyzed": layers,
            "successful_analyses": 0,
            "failed_analyses": 0,
            "unified_results": [],
            "feature_summaries": {}
        }
        
        progress = ProgressLogger(len(feature_files), "Batch analysis")
        
        for feature_file in feature_files:
            try:
                feature_results = self.analyze_feature(feature_file, layers, top_k)
                
                feature_name = os.path.splitext(os.path.basename(feature_file))[0]
                output_file = os.path.join(output_dir, f"{feature_name}_analysis.json")
                
                from .utils import save_json_results
                save_json_results(feature_results, output_file)
                
                batch_results["unified_results"].extend(feature_results["unified_results"])
                batch_results["feature_summaries"][feature_name] = {
                    "total_examples": feature_results["total_examples"],
                    "layers_completed": len([l for l in layers if l in feature_results["layer_results"] 
                                           and "error" not in feature_results["layer_results"][l]])
                }
                
                batch_results["successful_analyses"] += 1
                progress.update()
                
            except Exception as e:
                print(f"⚠️ Failed to analyze {feature_file}: {e}")
                batch_results["failed_analyses"] += 1
                progress.update()
        
        progress.finish()
        
        summary_file = os.path.join(output_dir, "batch_summary.json")
        save_json_results(batch_results, summary_file)
        
        return batch_results
    
    def clear_cache(self):
        self._model_cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()