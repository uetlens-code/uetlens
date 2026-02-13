# UETLens — Quick Usage Guide

This repo accompanies the paper "**UETLens: Understanding Event Types in Large Language Models**". It provides code to discover and analyze event type features in LLMs using SAEs (feature identification, intra-/inter-type overlap analysis, feature-based intervention, and training utilities).

## Installation

```bash
git clone https://github.com/uetlens-code/uetlens.git
cd uetlens
conda create -n uetlens python=3.12 -y && conda activate uetlens
pip install -r requirements.txt
pip install -e .
```

### Prereqs

- A base LLM (e.g., Llama-3.1-8B) available via transformers.
- Per-layer **OpenSAE** checkpoints, organized as:
  ```
  /path/to/sae/layer_{:02d}   # layer_00 ... layer_31
  ```

---

## Data you'll need

The repository includes pre-processed counterfactual datasets for 13 event subtypes under 7 coarse types, constructed from ACE 2005 and MAVEN:

| Coarse Type | ACE13 | MAVEN13 | MAVEN13-Large |
|-------------|-------|---------|---------------|
| Conflict    | 100   | 100     | 1,000 |
| Movement    | 100   | 100     | 1,000 |
| Life        | 200   | 200     | 2,000 |
| Contact     | 200   | 200     | 2,000 |
| Personnel   | 300   | 300     | 3,000 |
| Transaction | 200   | 200     | 1,325 |
| Justice     | 200   | 200     | 1,992 |
| **Total**   | **1,300** | **1,300** | **12,217** |

###  ACE 2005 Dataset Access 
Due to copyright restrictions, the ACE 2005 Multilingual Training Corpus cannot be distributed with this repository. You must obtain it directly from the Linguistic Data Consortium (LDC):  
- **Official source**: LDC2006T06 (https://doi.org/10.35111/mwxc-vh88)  
- **Access**: LDC members download directly; non-members must purchase a license.  



### File format

```
<event type="Attack">Britain has deployed some 45,000 troops to fight with the more than 250,000 US soldiers lined up against Iraqi troops</event>
<event type="Social_event">Britain has deployed some 45,000 troops to fight with the more than 250,000 US soldiers lined up against Iraqi troops</event>
```

Each `.txt` file follows alternating lines: odd = target event type, even = counterfactual with altered type attribute. All datasets are organized under `data/` by dataset name and subtype.



---

## Minimal examples

### A) Analyze an event type feature across layers (PS/PN/FRC)

```python
from uetlens.analyzer import Analyzer
from uetlens.utils import validate_layer_indices

model_path = "/path/to/llama-3.1-8b"
sae_tpl    = "/path/to/sae/layer_{:02d}"
feature    = "data/ACE13/Conflict_Attack.txt"
layers     = validate_layer_indices([0, 8, 15, 24, 30])

an = Analyzer(model_path, sae_tpl)
res = an.analyze_feature(feature_file=feature, layers=layers, top_k=10)

print(res["layer_results"][15]["top_features"][:5])
```

### B) Cross-type overlap analysis (intra- vs inter-type)

```python
from uetlens.overlap import OverlapAnalyzer

oa = OverlapAnalyzer(
    model_path="/path/to/Meta-Llama-3.1-8B",
    sae_tpl="/path/to/sae/OpenSAE-LLaMA-3.1-Layer_{:02d}"
)

coarse_mapping = {
    "Contact": ["Meet", "Phone-Write"],
    "Justice": ["Trial-Hearing", "Charge-Indict"],
    "Life": ["Die", "Injure"],
    "Conflict": ["Attack"],
    "Movement": ["Transport"],
    "Personnel": ["Elect", "Start-Position", "End-Position"],
    "Transaction": ["Transfer-Ownership", "Transfer-Money"]
}

similarity, subtypes, subtype_to_group = oa.compute_type_similarity(
    feature_dir="data/feature_sets",
    coarse_mapping=coarse_mapping,
    layers=[0, 8, 15, 24, 30],
    top_k=500
)

oa.plot_intra_inter_comparison(
    similarity,
    coarse_mapping,
    output_path="out/overlap/intra_inter_trend.png"
)
```

### C) Feature-based intervention (ablation/enhancement)

```python
from uetlens.intervener import Intervener

iv = Intervener(
    model_path="/path/to/llama-3.1-8b",
    sae_path="/path/to/sae/layer_00",
    use_multi_gpu=True
)

iv.run_intervention_experiment(
    input_prompt='''Event Type Classification Task: Classify whether the marked event is an attack event.

Event types: [attack, none]

Examples:
Sentence: "<event>As Kienmayer's columns fled to the east, they joined with elements of the Russian Empire's army in a rear guard action at the Battle of Amstetten on 5 November.</event>" Event type: none
Sentence: "<event>On 15 June, the Pakistani military intensified air strikes in North Waziristan and bombed eight foreign militant hideouts.</event>" Event type: attack

Now analyze this sentence:
Sentence: "<event>Utah, along with Sword on the eastern flank, was added to the invasion plan in December 1943.</event>" Event type:''',
    intervention_indices=[101989],
    output_path="out/intervene/event_attack_L00.txt",
    num_generations=10,
    max_new_tokens=1,
    temperature=0.7,
    experiment_name="event_attack_L0"
)
```

### D) Train an SVM classifier on event type features

```python
from uetlens.svm import EventTypeSVMClassifier

model_path = "/path/to/llama-3.1-8b"
sae_path_template = "/path/to/sae/layer_{:02d}"

classifier = EventTypeSVMClassifier(
    model_path=model_path,
    sae_path_template=sae_path_template,
    cuda_devices="0,1,2,3"
)

training_data = [
    ('<event>The army launched an attack on the rebel stronghold</event>', 'Attack'),
    ('<event>Militants bombed the government building yesterday</event>', 'Attack'),
    ('<event>The truck transported goods across the border</event>', 'Transport'),
    ('<event>Five people died in the car accident</event>', 'Die'),
    ('<event>Several civilians were injured in the explosion</event>', 'Injure'),
    ('<event>Leaders from both countries met to discuss peace</event>', 'Meet'),
    ('<event>The committee elected a new chairman</event>', 'Elect'),
    ('<event>The court held a hearing on the case</event>', 'Trial')
]

classifier.train_svm(training_data, output_dir="out/svm_event_type")

test_sentences = [
    '<event>Rebels attacked the military base at dawn</event>',
    '<event>The convoy moved through the desert</event>',
    '<event>Three people lost their lives in the fire</event>',
    '<event>Many were wounded in the explosion</event>',
    '<event>Diplomats met to negotiate the treaty</event>',
    '<event>The committee elected new leadership</event>',
    '<event>The court hearing lasted all day</event>'
]

true_labels = ['Attack', 'Transport', 'Die', 'Injure', 'Meet', 'Elect', 'Trial']

predictions, probabilities = classifier.predict(test_sentences, true_labels, output_dir="out/svm_event_type")
print(f"Predictions: {predictions}")
```

---

## Tips

- **GPU strongly recommended**; device is auto-detected (override via class `device` args).
- **Ensure feature files are non-empty** and follow the **alternating pair format**; otherwise PS/PN/FRC scores will be degenerate.
- **For overlap analysis**, pre-extracted feature sets can be placed in `data/feature_sets/` to skip feature extraction.