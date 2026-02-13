import os
from typing import List, Dict, Any, Optional
import torch
import transformers
from opensae.transformer_with_sae import TransformerWithSae
from opensae import OpenSae

from .metrics import compute_layer_stats, get_top_features_by_frc
from .utils import load_text_data, setup_tokenizer, get_available_device


class Visualizer:
    def __init__(
        self,
        model_path: str,
        sae_path_template: str = "/path/to/sae/layer_{:02d}",
        device: Optional[str] = None
    ):
        self.model_path = model_path
        self.sae_path_template = sae_path_template
        self.device = device or get_available_device()
        self.tokenizer = setup_tokenizer(model_path)
    
    def generate_html_report(
        self,
        feature_file: str,
        layer_idx: int,
        output_html: str,
        top_k: int = 10,
        manual_base_vectors: Optional[List[int]] = None,
        analysis_mode: str = "FRC"
    ) -> Dict[str, Any]:
        sae_path = self.sae_path_template.format(layer_idx)
        if not os.path.exists(sae_path):
            raise FileNotFoundError(f"SAE model not found: {sae_path}")
        
        sae = OpenSae.from_pretrained(sae_path)
        model = TransformerWithSae(self.model_path, sae, self.device)
        
        lines = load_text_data(feature_file)
        
        print(f"[INPUT TEXT] {len(lines)} examples from {feature_file}")
        
        encodings = self.tokenizer(
            lines,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)
        
        structured_data = model.extract_data(encodings, self.tokenizer)
        
        if analysis_mode == "FRC":
            analysis_results = self._analyze_frc_mode(structured_data, top_k)
        else:
            analysis_results = self._analyze_frequency_mode(structured_data, top_k)
        
        base_vector_indices = []
        if "top_k_results" in analysis_results and analysis_results["top_k_results"]:
            base_vector_indices = [result[0] for result in analysis_results["top_k_results"]]
        
        if manual_base_vectors:
            base_vector_indices.extend(manual_base_vectors)
        
        base_vector_indices = list(set(base_vector_indices))
        
        if base_vector_indices:
            model.visualize(structured_data, base_vector_indices, output_html=output_html)
            print(f"[VISUALIZE] Visualization saved to '{output_html}'")
        else:
            print("[VISUALIZE] No valid base vectors to visualize.")
        
        results = {
            "feature_file": feature_file,
            "layer": layer_idx,
            "analysis_mode": analysis_mode,
            "total_examples": len(lines),
            "base_vectors_visualized": base_vector_indices,
            "output_html": output_html,
            "analysis_results": analysis_results
        }
        
        return results
    
    def _analyze_frc_mode(self, structured_data: List[Dict], top_k: int) -> Dict[str, Any]:
        layer_stats = compute_layer_stats(structured_data)
        top_features = get_top_features_by_frc(layer_stats, top_k)
        
        return {
            "mode": "FRC",
            "top_k_results": top_features,
            "full_stats": layer_stats,
            "total_base_vectors": len(layer_stats)
        }
    
    def _analyze_frequency_mode(self, structured_data: List[Dict], top_k: int) -> Dict[str, Any]:
        base_vector_counts = {}
        for sentence in structured_data:
            for token in sentence["tokens"]:
                for activation in token["activations"]:
                    base_vec = activation["base_vector"]
                    base_vector_counts[base_vec] = base_vector_counts.get(base_vec, 0) + 1
        
        frequency_results = sorted(
            base_vector_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        return {
            "mode": "frequency",
            "top_k_results": frequency_results,
            "total_base_vectors": len(base_vector_counts),
            "frequency_distribution": dict(frequency_results)
        }
    
    def batch_visualize(
        self,
        feature_files: List[str],
        layer_idx: int,
        output_dir: str,
        top_k: int = 10,
        analysis_mode: str = "FRC"
    ) -> Dict[str, Any]:
        os.makedirs(output_dir, exist_ok=True)
        
        batch_results = {
            "layer": layer_idx,
            "analysis_mode": analysis_mode,
            "total_features": len(feature_files),
            "successful_visualizations": 0,
            "failed_visualizations": 0,
            "feature_results": {}
        }
        
        for feature_file in feature_files:
            try:
                feature_name = os.path.splitext(os.path.basename(feature_file))[0]
                output_html = os.path.join(output_dir, f"{feature_name}.html")
                
                results = self.generate_html_report(
                    feature_file,
                    layer_idx,
                    output_html,
                    top_k,
                    analysis_mode=analysis_mode
                )
                
                batch_results["feature_results"][feature_name] = results
                batch_results["successful_visualizations"] += 1
                
                print(f"Generated visualization for {feature_name}")
                
            except Exception as e:
                print(f"⚠️ Failed to visualize {feature_file}: {e}")
                batch_results["failed_visualizations"] += 1
        
        summary_file = os.path.join(output_dir, "visualization_summary.json")
        from .utils import save_json_results
        save_json_results(batch_results, summary_file)
        
        print(f"\n Batch visualization complete. Summary saved to {summary_file}")
        
        return batch_results
    
    def compare_layers_visualization(
        self,
        feature_file: str,
        layers: List[int],
        output_dir: str,
        top_k: int = 10
    ) -> Dict[str, Any]:
        os.makedirs(output_dir, exist_ok=True)
        
        feature_name = os.path.splitext(os.path.basename(feature_file))[0]
        
        comparison_results = {
            "feature": feature_name,
            "layers": layers,
            "layer_visualizations": {},
            "cross_layer_summary": {}
        }
        
        for layer_idx in layers:
            try:
                output_html = os.path.join(output_dir, f"{feature_name}_layer_{layer_idx:02d}.html")
                
                results = self.generate_html_report(
                    feature_file,
                    layer_idx,
                    output_html,
                    top_k
                )
                
                comparison_results["layer_visualizations"][layer_idx] = results
                print(f"Generated layer {layer_idx} visualization")
                
            except Exception as e:
                print(f"Failed layer {layer_idx}: {e}")
        
        summary_file = os.path.join(output_dir, f"{feature_name}_layer_comparison.json")
        from .utils import save_json_results
        save_json_results(comparison_results, summary_file)
        
        return comparison_results