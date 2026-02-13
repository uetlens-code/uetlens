from dataclasses import dataclass
from simple_parsing import Serializable, list_field, field

from multiprocessing import cpu_count

@dataclass
class SaeConfig(Serializable):

    expansion_factor: int = 32

    normalize_decoder: bool = True

    num_latents: int = 0

    k: int = 32
    
    multi_topk: bool = False
    
    l1_coef: float | None = None
    
    input_normalize: bool = True
    
    shift_back: bool = False

@dataclass
class TrainConfig(Serializable):
    seed: int = 42

    dp_size: int = 1
    
    fsdp: bool = False
    
    mp_size: int = 1

    local_batch_size: int = 8
    
    global_batch_size: int = 512

    grad_acc_steps: int | None = None

    micro_acc_steps: int = 1

    lr: float | None = None
    
    min_lr_ratio: float = 0.02

    lr_scheduler_type: str = "linear"

    lr_warmup_ratio: float | None = None

    lr_warmup_steps: int | None = None
    
    lr_stable_steps: int | None = None
    
    lr_decay_ratio: float | None = None
    
    lr_decay_steps: int | None = None
    
    auxk_alpha: float = 0.0

    dead_feature_threshold: int = 10_000_000

    hookpoint: str | None = None

    distribute_modules: bool = False

    save_every: int = 1000
    
    save_at_init: bool = False
    
    save_dir: str = "output/ckpts"
    
    load_dir: str = None
    
    adam_in_8bit: bool = False
    
    spike_detection_start: int = 10
    
    spike_detection_window_size: int = 5
    
    spike_detection_threshold_ratio: float = 1.05
    
    varlen: bool = True
    
    k_scheduler: str = "constant"
    
    k_scheduler_factor: int = 4
    
    k_scheduler_step_ratio: float = 0.25
    
    early_exit_inference_layer_num: int | None = None

    log_to_wandb: bool = True
    run_name: str | None = None
    wandb_project: str | None = None
    wandb_id: str | None = None
    wandb_log_frequency: int = 1

    def __post_init__(self):
        assert not (
            self.grad_acc_steps and self.global_batch_size
        ), "Cannot specify both `grad_acc_steps` and `global_batch_size`."
        
        if self.global_batch_size:
            assert self.global_batch_size % (self.local_batch_size * self.dp_size) == 0, "Global batch size must be divisible by local batch size * dp_size"
            self.grad_acc_steps = self.global_batch_size // (self.local_batch_size * self.dp_size)
            
        if self.run_name is None:
            self.run_name = "default_run"
        if self.wandb_project is None:
            self.wandb_project = self.run_name
        if self.wandb_id is None:
            self.wandb_id  = self.run_name
            
        assert not (self.lr_warmup_steps and self.lr_warmup_ratio), "Cannot specify both `lr_warmup_steps` and `lr_warmup_ratio`."
        
        assert self.spike_detection_window_size <= self.spike_detection_start
        
        assert self.k_scheduler in ["constant", "linear", "piecewise", "random", "discrete_linear"]
        
        assert self.k_scheduler_step_ratio > 0 and self.k_scheduler_step_ratio < 1
        
        assert self.lr_scheduler_type in ["linear", "cosine", "wsd"]

@dataclass
class ModelConfig(Serializable):

    model: str = "EleutherAI/pythia-160m"
    
    auto_model_class: str = "AutoModel"
    
    trust_remote_code: bool = True

    load_in_8bit: bool = False

    def __post_init__(self):
        assert self.auto_model_class in ["AutoModel", "AutoModelForCausalLM"]
        if self.auto_model_class == "AutoModel":
            from transformers import AutoModel
            self.auto_model_class = AutoModel
        else:
            from transformers import AutoModelForCausalLM
            self.auto_model_class = AutoModelForCausalLM

@dataclass
class DataConfig(Serializable):
    
    dataset: str = "togethercomputer/RedPajama-Data-1T-Sample"

    split: str = "train"

    data_preprocessing_num_proc: int = field(
        default_factory=lambda: cpu_count() // 2,
    )
    
    trust_remote_code: bool = True
    
    ctx_len: int = 2048