import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

class OverlapAnalyzer:
    def __init__(self, model_path, sae_tpl):
        self.model_path = model_path
        self.sae_tpl = sae_tpl
    
    def _load_vector_ids(self, file_path, layers=[0,8,15,24,30]):
        vector_ids = {layer: [] for layer in layers}
        with open(file_path, 'r') as f:
            lines = f.readlines()
        for i, layer in enumerate(layers):
            if i < len(lines):
                vector_ids[layer] = [x.strip() for x in lines[i].strip().split(',') if x.strip()]
        return vector_ids
    
    def _format_filename(self, subtype, coarse_mapping):
        for group, subtypes in coarse_mapping.items():
            if subtype in subtypes:
                if group == "Contact":
                    if subtype == "Meet":
                        return "Contact_Meet_vector_ids.txt"
                    elif subtype == "Phone-Write":
                        return "Contact_Phone_Write_vector_ids.txt"
                elif group == "Justice":
                    if subtype == "Trial-Hearing":
                        return "Justice_Trial_Hearing_vector_ids.txt"
                    elif subtype == "Charge-Indict":
                        return "Justice_Charge_Indict_vector_ids.txt"
                elif group == "Life":
                    if subtype == "Die":
                        return "Life_Die_vector_ids.txt"
                    elif subtype == "Injure":
                        return "Life_Injure_vector_ids.txt"
                elif group == "Conflict":
                    if subtype == "Attack":
                        return "Conflict_Attack_vector_ids.txt"
                elif group == "Movement":
                    if subtype == "Transport":
                        return "Movement_Transport_vector_ids.txt"
                elif group == "Personnel":
                    if subtype == "Elect":
                        return "Personnel_Elect_vector_ids.txt"
                    elif subtype == "Start-Position":
                        return "Personnel_Start_Position_vector_ids.txt"
                    elif subtype == "End-Position":
                        return "Personnel_End_Position_vector_ids.txt"
                elif group == "Transaction":
                    if subtype == "Transfer-Ownership":
                        return "Transaction_Transfer_Ownership_vector_ids.txt"
                    elif subtype == "Transfer-Money":
                        return "Transaction_Transfer_Money_vector_ids.txt"
        return f"{subtype}_vector_ids.txt"
    
    def compute_type_similarity(self, feature_dir, coarse_mapping, layers=[0,8,15,24,30], top_k=500):
        all_subtypes = []
        for group, subtypes in coarse_mapping.items():
            all_subtypes.extend(subtypes)
        
        subtype_to_group = {}
        for group, subtypes in coarse_mapping.items():
            for subtype in subtypes:
                subtype_to_group[subtype] = group
        
        vector_ids_by_type = {}
        missing_types = []
        for subtype in all_subtypes:
            filename = self._format_filename(subtype, coarse_mapping)
            file_path = os.path.join(feature_dir, filename)
            if os.path.exists(file_path):
                vector_ids_by_type[subtype] = self._load_vector_ids(file_path, layers)
                print(f"Loaded: {filename}")
            else:
                missing_types.append(subtype)
                print(f"Warning: {filename} not found, skipping...")
        
        existing_subtypes = [st for st in all_subtypes if st in vector_ids_by_type]
        
        similarity = {layer: np.zeros((len(existing_subtypes), len(existing_subtypes))) for layer in layers}
        
        for layer in layers:
            for i, t1 in enumerate(existing_subtypes):
                for j, t2 in enumerate(existing_subtypes):
                    if i <= j:
                        set1 = set(vector_ids_by_type[t1][layer][:top_k])
                        set2 = set(vector_ids_by_type[t2][layer][:top_k])
                        overlap = len(set1 & set2) / max(len(set1), len(set2))
                        similarity[layer][i, j] = overlap
                        similarity[layer][j, i] = overlap
        
        if missing_types:
            print(f"Missing types: {missing_types}")
        
        return similarity, existing_subtypes, subtype_to_group
    
    def plot_intra_inter_comparison(self, similarity, coarse_mapping, output_path, existing_subtypes=None):
        layers = sorted(similarity.keys())
        
        if existing_subtypes is None:
            n_subtypes = similarity[layers[0]].shape[0]
            all_subtypes = []
            for group, subtypes in coarse_mapping.items():
                all_subtypes.extend(subtypes)
            existing_subtypes = all_subtypes[:n_subtypes]
        
        subtype_to_group = {}
        for group, subtypes in coarse_mapping.items():
            for subtype in subtypes:
                subtype_to_group[subtype] = group
        
        intra_avgs = []
        inter_avgs = []
        
        for layer in layers:
            sim = similarity[layer]
            intra_scores = []
            inter_scores = []
            
            for i in range(len(existing_subtypes)):
                for j in range(i+1, len(existing_subtypes)):
                    if i < sim.shape[0] and j < sim.shape[1]:
                        t1, t2 = existing_subtypes[i], existing_subtypes[j]
                        score = sim[i, j]
                        
                        if subtype_to_group[t1] == subtype_to_group[t2]:
                            intra_scores.append(score)
                        else:
                            inter_scores.append(score)
            
            intra_avgs.append(np.mean(intra_scores) if intra_scores else 0)
            inter_avgs.append(np.mean(inter_scores) if inter_scores else 0)
        
        plt.figure(figsize=(10, 6))
        plt.plot(layers, intra_avgs, marker='o', label='Intra-Type', linewidth=2)
        plt.plot(layers, inter_avgs, marker='s', label='Inter-Type', linewidth=2)
        plt.xlabel('Layer')
        plt.ylabel('Overlap Similarity')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Plot saved to: {output_path}")
        
        return intra_avgs, inter_avgs