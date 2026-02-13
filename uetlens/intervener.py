import os
from typing import List, Dict, Any, Optional, Union
import torch
import transformers
from opensae.transformer_with_sae import TransformerWithSae, InterventionConfig
from opensae import OpenSae

from .utils import setup_tokenizer, get_available_device, ProgressLogger

class Intervener:
    def __init__(
        self,
        model_path: str,
        sae_path: str,
        device: Optional[str] = None,
        use_multi_gpu: bool = False
    ):
        self.model_path = model_path
        self.sae_path = sae_path
        self.device = device or get_available_device()
        self.use_multi_gpu = use_multi_gpu
        self.tokenizer = setup_tokenizer(model_path)
        self._load_models()
    
    def _load_models(self):
        if not os.path.exists(self.sae_path):
            raise FileNotFoundError(f"SAE model not found: {self.sae_path}")
        
        print(f"Loading SAE from {self.sae_path}")
        sae = OpenSae.from_pretrained(self.sae_path)
        
        print(f"Loading transformer model from {self.model_path}")
        self.model = TransformerWithSae(
            self.model_path, 
            sae, 
            self.device,
            use_multi_gpu=self.use_multi_gpu
        )
        
        print("Models loaded successfully!")
    
    def run_intervention_experiment(
        self,
        input_prompt: str,
        intervention_indices: List[int],
        output_path: str,
        num_generations: int = 10,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        experiment_name: Optional[str] = None
    ) -> Dict[str, Any]:
        experiment_results = {
            "experiment_name": experiment_name or "intervention_experiment",
            "intervention_indices": intervention_indices,
            "num_generations": num_generations,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "conditions": {
                "ablation": [],
                "enhancement": [],
                "control": []
            }
        }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as out_file:
            out_file.write(f"Experiment: {experiment_results['experiment_name']}\n\n")
            out_file.write(f"Intervention indices: {intervention_indices}\n\n")
        
        progress = ProgressLogger(num_generations * 3, "Running intervention experiment")
        
        for generation in range(1, num_generations + 1):
            print(f"Starting generation {generation}...")
            
            inputs = self.tokenizer(input_prompt, return_tensors="pt").to(self.device)
            
            generation_results = {
                "generation": generation,
                "ablation": None,
                "enhancement": None, 
                "control": None
            }
            
            with open(output_path, "a", encoding="utf-8") as out_file:
                out_file.write(f"--- Generation {generation} ---\n")
                
                ablation_config = InterventionConfig(
                    intervention=True,
                    intervention_mode="set",
                    intervention_indices=intervention_indices,
                    intervention_value=0.0,
                    prompt_only=False,
                )
                self.model.update_intervention_config(ablation_config)
                
                ablation_output = self._generate_with_intervention(
                    inputs, max_new_tokens, temperature
                )
                generated_part = self._extract_generated_part(input_prompt, ablation_output)
                generation_results["ablation"] = generated_part
                
                out_file.write("[Ablation]:\n")
                out_file.write(generated_part + "\n\n")
                progress.update()
                
                enhancement_config = InterventionConfig(
                    intervention=True,
                    intervention_mode="set",
                    intervention_indices=intervention_indices,
                    intervention_value=2.0,
                    prompt_only=False,
                )
                self.model.update_intervention_config(enhancement_config)
                
                enhancement_output = self._generate_with_intervention(
                    inputs, max_new_tokens, temperature
                )
                generated_part = self._extract_generated_part(input_prompt, enhancement_output)
                generation_results["enhancement"] = generated_part
                
                out_file.write("[Enhancement]:\n")
                out_file.write(generated_part + "\n\n")
                progress.update()
                
                control_config = InterventionConfig(
                    intervention=True,
                    intervention_mode="multiply",
                    intervention_indices=intervention_indices,
                    intervention_value=1.0,
                    prompt_only=False,
                )
                self.model.update_intervention_config(control_config)
                
                control_output = self._generate_with_intervention(
                    inputs, max_new_tokens, temperature
                )
                generated_part = self._extract_generated_part(input_prompt, control_output)
                generation_results["control"] = generated_part
                
                out_file.write("[Control]:\n")
                out_file.write(generated_part + "\n\n")
                progress.update()
            
            experiment_results["conditions"]["ablation"].append(generation_results["ablation"])
            experiment_results["conditions"]["enhancement"].append(generation_results["enhancement"])
            experiment_results["conditions"]["control"].append(generation_results["control"])
            
            print(f"Generation {generation} completed and written to {output_path}.")
        
        progress.finish()
        
        results_json_path = output_path.replace('.txt', '_results.json')
        from .utils import save_json_results
        save_json_results(experiment_results, results_json_path)
        
        print(f"\n✅ Experiment complete! Results saved to {output_path}")
        print(f"Structured results saved to {results_json_path}")
        
        return experiment_results
    
    def _extract_generated_part(self, input_prompt: str, full_output: str) -> str:
        if full_output.startswith(input_prompt):
            return full_output[len(input_prompt):].strip()
        return full_output.strip()
    
    def _generate_with_intervention(
        self,
        inputs: Dict[str, torch.Tensor],
        max_new_tokens: int,
        temperature: float
    ) -> str:
        try:
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True if temperature > 0 else False,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            output_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            return output_text
            
        except Exception as e:
            return f"[GENERATION ERROR]: {str(e)}"
    
    def batch_intervention_experiments(
        self,
        input_files: List[str],
        intervention_configs: List[Dict[str, Any]],
        output_dir: str,
        num_generations: int = 5
    ) -> Dict[str, Any]:
        os.makedirs(output_dir, exist_ok=True)
        
        batch_results = {
            "total_experiments": len(input_files) * len(intervention_configs),
            "successful_experiments": 0,
            "failed_experiments": 0,
            "experiment_summaries": {}
        }
        
        progress = ProgressLogger(
            len(input_files) * len(intervention_configs),
            "Batch intervention experiments"
        )
        
        for input_file in input_files:
            with open(input_file, "r", encoding="utf-8") as f:
                input_prompt = f.read().strip()
            
            input_name = os.path.splitext(os.path.basename(input_file))[0]
            
            for config in intervention_configs:
                try:
                    experiment_name = f"{input_name}_{config['name']}"
                    output_path = os.path.join(output_dir, f"{experiment_name}.txt")
                    
                    results = self.run_intervention_experiment(
                        input_prompt=input_prompt,
                        intervention_indices=config["intervention_indices"],
                        output_path=output_path,
                        num_generations=num_generations,
                        max_new_tokens=config.get("max_new_tokens", 100),
                        temperature=config.get("temperature", 1.0),
                        experiment_name=experiment_name
                    )
                    
                    batch_results["experiment_summaries"][experiment_name] = {
                        "input_file": input_file,
                        "config": config,
                        "output_path": output_path,
                        "status": "success"
                    }
                    
                    batch_results["successful_experiments"] += 1
                    progress.update()
                    
                except Exception as e:
                    print(f"⚠️ Failed experiment {experiment_name}: {e}")
                    
                    batch_results["experiment_summaries"][experiment_name] = {
                        "input_file": input_file,
                        "config": config,
                        "status": "failed",
                        "error": str(e)
                    }
                    
                    batch_results["failed_experiments"] += 1
                    progress.update()
        
        progress.finish()
        
        summary_path = os.path.join(output_dir, "batch_intervention_summary.json")
        from .utils import save_json_results
        save_json_results(batch_results, summary_path)
        
        print(f"\nBatch intervention complete! Summary saved to {summary_path}")
        
        return batch_results
    
    def quick_intervention_test(
        self,
        input_prompt: str,
        intervention_indices: List[int],
        intervention_values: List[float] = [0.0, 1.0, 5.0, 10.0]
    ) -> Dict[str, str]:
        results = {}
        
        inputs = self.tokenizer(input_prompt, return_tensors="pt").to(self.device)
        
        for value in intervention_values:
            config = InterventionConfig(
                intervention=True,
                intervention_mode="set",
                intervention_indices=intervention_indices,
                intervention_value=value,
                prompt_only=False,
            )
            
            self.model.update_intervention_config(config)
            
            output = self._generate_with_intervention(inputs, max_new_tokens=50, temperature=0.7)
            generated_part = self._extract_generated_part(input_prompt, output)
            results[f"value_{value}"] = generated_part
            
            print(f"Intervention value {value}: {generated_part[:100]}...")
        
        return results