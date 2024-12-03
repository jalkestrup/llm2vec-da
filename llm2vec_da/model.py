from typing import Optional, List

import torch
#from llm2vec.models import MistralBiForMNTP, LlamaBiForMNTP
from .model_modifications.bidirectional_llama import LlamaBiForMNTP
from peft import LoraConfig, get_peft_model
from transformers import AutoConfig

from llm2vec_da.arguments import ModelArguments


def get_model_class(config):
    config_class_name = config.__class__.__name__
    if config_class_name == "LlamaConfig":
        return LlamaBiForMNTP
    else:
        raise ValueError(f"Model class {config_class_name} not supported.")

def initialize_peft(
    model,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_modules: Optional[List[str]] = None,
):
    if lora_modules is None and model.config.__class__.__name__ in [
        "LlamaConfig",
        "MistralConfig",
    ]:
        lora_modules = [
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    elif lora_modules is None:
        raise ValueError("lora_modules must be specified for this model.")

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=None,
    )
    # model organization is MODEL_TYPEBiForMNTP.model -> MODEL_TYPELBiModel, we have to apply PEFT to the inner model
    peft_model = get_peft_model(model, config)
    print(f"Model's Lora trainable parameters:")
    peft_model.print_trainable_parameters()
    return peft_model




def load_adapted_model(model_args: ModelArguments, adapter_path):
    """
    Load a model with an adapter from a given path. Note that saving the model does not persist the lm_head, so
    the adapted model is not suitable for generation tasks.
    """
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }
    config = None
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path, **config_kwargs
        )
    model_class = get_model_class(config)
    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    model = model_class.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=model_args.low_cpu_mem_usage,
        attn_implementation=model_args.attn_implementation
    )
    model.model.load_adapter(adapter_path)
    return model

#from llm2vec.loss.utils import load_loss
