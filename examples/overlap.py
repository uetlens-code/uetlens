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