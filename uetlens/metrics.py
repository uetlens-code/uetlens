from typing import Dict, List, Any, Tuple
import torch
from collections import defaultdict

def compute_ps_pn_frc(structured_data: List[Dict], base_vector: int) -> Tuple[float, float, float]:
    pos_sents = [s for s in structured_data if s["sentence_id"] % 2 == 0]
    neg_sents = [s for s in structured_data if s["sentence_id"] % 2 == 1]
    
    num_pos = len(pos_sents)
    num_neg = len(neg_sents)
    
    if num_pos == 0 or num_neg == 0:
        return 0.0, 0.0, 0.0
    
    ps = sum(
        1 for s in pos_sents
        if any(act["base_vector"] == base_vector 
               for t in s["tokens"] 
               for act in t["activations"])
    ) / num_pos
    
    pn = sum(
        1 for s in neg_sents
        if not any(act["base_vector"] == base_vector
                   for t in s["tokens"]
                   for act in t["activations"])
    ) / num_neg
    
    frc = 2 * ps * pn / (ps + pn) if (ps + pn) > 0 else 0.0
    
    return ps, pn, frc

def compute_layer_stats(structured_data: List[Dict]) -> Dict[int, Dict[str, float]]:
    pos_sents = [s for s in structured_data if s["sentence_id"] % 2 == 0]
    neg_sents = [s for s in structured_data if s["sentence_id"] % 2 == 1]
    
    num_pos, num_neg = len(pos_sents), len(neg_sents)
    
    if num_pos == 0 or num_neg == 0:
        return {}
    
    base_to_pos_presence = defaultdict(int)
    base_to_neg_presence = defaultdict(int)
    base_to_max_acts = defaultdict(list)
    
    for s in structured_data:
        is_pos = s["sentence_id"] % 2 == 0
        sentence_has_base = defaultdict(bool)
        
        for t in s["tokens"]:
            for act in t["activations"]:
                b = act["base_vector"]
                sentence_has_base[b] = True
                if is_pos:
                    base_to_max_acts[b].append(act["activation"])
        
        for b in sentence_has_base:
            if is_pos:
                base_to_pos_presence[b] += 1
            else:
                base_to_neg_presence[b] += 1
    
    all_bases = set(list(base_to_pos_presence.keys()) + list(base_to_neg_presence.keys()))
    
    stats = {}
    for b in all_bases:
        ps = base_to_pos_presence.get(b, 0) / num_pos
        pn = (num_neg - base_to_neg_presence.get(b, 0)) / num_neg
        frc = 2 * ps * pn / (ps + pn) if (ps + pn) > 0 else 0.0
        
        max_acts = base_to_max_acts.get(b, [])
        avg_max = max(max_acts) if max_acts else 0.0
        
        stats[b] = {
            "ps": ps,
            "pn": pn,
            "frc": frc,
            "avg_max_activation": avg_max
        }
    
    return stats

def get_top_features_by_frc(stats: Dict[int, Dict[str, float]], top_k: int = 10) -> List[Tuple[int, float]]:
    frc_list = [(base_vec, stat_dict["frc"]) for base_vec, stat_dict in stats.items()]
    frc_list.sort(key=lambda x: x[1], reverse=True)
    return frc_list[:top_k]

def get_top_features_by_ps(stats: Dict[int, Dict[str, float]], top_k: int = 10) -> List[Tuple[int, float]]:
    ps_list = [(base_vec, stat_dict["ps"]) for base_vec, stat_dict in stats.items()]
    ps_list.sort(key=lambda x: x[1], reverse=True)
    return ps_list[:top_k]

def get_top_features_by_pn(stats: Dict[int, Dict[str, float]], top_k: int = 10) -> List[Tuple[int, float]]:
    pn_list = [(base_vec, stat_dict["pn"]) for base_vec, stat_dict in stats.items()]
    pn_list.sort(key=lambda x: x[1], reverse=True)
    return pn_list[:top_k]

def get_top_features_by_activation(stats: Dict[int, Dict[str, float]], top_k: int = 10) -> List[Tuple[int, float]]:
    act_list = [(base_vec, stat_dict["avg_max_activation"]) for base_vec, stat_dict in stats.items()]
    act_list.sort(key=lambda x: x[1], reverse=True)
    return act_list[:top_k]