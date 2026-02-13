from uetlens.analyzer import Analyzer
from uetlens.utils import validate_layer_indices

model_path = "/path/to/Meta-Llama-3.1-8B"
sae_tpl    = "/path/to/sae/OpenSAE-LLaMA-3.1-Layer_{:02d}"
feature    = "data/ACE13/Conflict_Attack.txt"
layers     = validate_layer_indices([15])

an = Analyzer(model_path, sae_tpl)
res = an.analyze_feature(feature_file=feature, layers=layers, top_k=10)

print(res["layer_results"][15]["top_features"][:5])