import datasets
import transformers
import os
import torch
import wandb
import argparse
from transformers import (
    HfArgumentParser,
    AutoConfig,
    AutoTokenizer,
    TrainingArguments,
    is_torch_tpu_available
)
from peft import LoraConfig, get_peft_model
from typing import List, Optional
from llm2vec_da.arguments import ModelArguments, MNTPDataTrainingArguments, CustomArguments
from llm2vec_da.model import get_model_class
from llm2vec_da.training import (
    DataCollatorForLanguageModelingWithFullMasking,
    MNTPTrainer,
    StopTrainingCallback
)
from llm2vec_da.metrics import MetricEvaluator, preprocess_logits_for_metrics
from dotenv import load_dotenv
from datasets import load_dataset

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
    peft_model = get_peft_model(model, config)
    print(f"Model's Lora trainable parameters:")
    peft_model.print_trainable_parameters()
    return peft_model


def parse_args():
    parser = argparse.ArgumentParser(description='MNTP Training Script')
    parser.add_argument(
        '--config', 
        type=str, 
        default="configs/mntp/MetaLlama3-sheared.json",
        help='Path to the configuration JSON file'
    )
    return parser.parse_args()


def main():
    # Parse command line arguments
    load_dotenv()
    args = parse_args()
    
    # Parse model arguments from config file
    parser = HfArgumentParser(
        (ModelArguments, MNTPDataTrainingArguments, TrainingArguments, CustomArguments)
    )
    
    model_args, data_args, training_args, custom_args = parser.parse_json_file(args.config)

    # Setup config kwargs and seed
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }
    
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}
    
    transformers.set_seed(training_args.seed)

    # Load config and get model class
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path, **config_kwargs
    )
    model_class = get_model_class(config)

    # Initialize model
    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    
    model = model_class.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        device_map="auto",
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=model_args.low_cpu_mem_usage,
        attn_implementation="flash_attention_2",  # Note: sdpa for running on L4/T4. Set back to flash_attention when running on A100 GPU
    )

    # Setup PEFT - first get PEFT-wrapped version of inner model
    peft_model = initialize_peft(
        model.model,  # Pass the inner transformer model
        lora_r=custom_args.lora_r,
        lora_alpha=2 * custom_args.lora_r,
        lora_dropout=custom_args.lora_dropout,
    )
    # Replace inner model with PEFT-wrapped version
    model.model = peft_model

    # Setup tokenizer
    tokenizer_kwargs = {
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, **tokenizer_kwargs
    )

    # Handle special tokens
    if tokenizer.mask_token is None:
        if custom_args.mask_token_type == "blank":
            print("Setting mask token to _")
            tokenizer.mask_token = "_"
        elif custom_args.mask_token_type == "eos":
            print("Setting mask token to eos")
            tokenizer.mask_token = tokenizer.eos_token
        elif custom_args.mask_token_type == "mask":
            print("Setting mask token to <mask>")
            tokenizer.add_tokens(["<mask>"])
            tokenizer.mask_token = "<mask>"
        else:
            raise ValueError(
                f"mask_token_type {custom_args.mask_token_type} is not supported."
            )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Setup data collator
    data_collator = DataCollatorForLanguageModelingWithFullMasking(
        tokenizer=tokenizer,
        mlm_probability=data_args.mlm_probability
    )

    # Load datasets : Local files
    # tokenized_datasets = datasets.load_from_disk(custom_args.tokenized_dataset_path)
    
    # Load datasets : Huggingface
    tokenized_datasets = load_dataset(custom_args.tokenized_dataset_path,
                                      split=None,
                                      use_auth_token=True)

    
    train_dataset = tokenized_datasets["train"]
    if data_args.max_train_samples is not None:
        max_train_samples = min(len(train_dataset), data_args.max_train_samples)
        train_dataset = train_dataset.select(range(max_train_samples))

    eval_dataset = tokenized_datasets["validation"]
    if data_args.max_eval_samples is not None:
        max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
        eval_dataset = eval_dataset.select(range(max_eval_samples))

    # Setup evaluator
    evaluator = MetricEvaluator(model_args.cache_dir)

    # Setup wandb
    os.environ["WANDB_PROJECT"] = custom_args.wandb_project
    os.environ["WANDB_LOG_MODEL"] = custom_args.wandb_log_model
    if custom_args.wandb_run_group:
        os.environ["WANDB_RUN_GROUP"] = custom_args.wandb_run_group
    if custom_args.wandb_watch:
        os.environ["WANDB_WATCH"] = custom_args.wandb_watch

    # Initialize trainer
    trainer = MNTPTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=evaluator if training_args.do_eval and not is_torch_tpu_available()
                              else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
        if training_args.do_eval and not is_torch_tpu_available()
        else None,
    )

    model.config.use_cache = False
    trainer.add_callback(StopTrainingCallback(custom_args.stop_after_n_steps))

    # Start training
    train_result = trainer.train()

    # Save final model and metrics
    trainer.save_model()
    metrics = train_result.metrics

    max_train_samples = (
        data_args.max_train_samples
        if data_args.max_train_samples is not None
        else len(train_dataset)
    )
    metrics["train_samples"] = min(max_train_samples, len(train_dataset))

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # Finish wandb run
    wandb.finish()
    
    import subprocess
    import time
    time.sleep(120)  # Wait 2 minutes to ensure all processes are properly saved
    subprocess.run(["sudo", "shutdown", "-h", "now"])



if __name__ == "__main__":
    main() 