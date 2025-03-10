import torch
from accelerate import Accelerator
import transformers
from tqdm import tqdm
import os
import argparse
from huggingface_hub import login
from transformers import HfArgumentParser
from llm2vec_da.arguments import (
    SimCSEModelArguments,
    SimCSEDataTrainingArguments,
    TrainingArguments,
    SimCSECustomArguments,
)
from llm2vec_da.data_utils import PairedDataset, _load_from_files
from llm2vec_da import LLM2Vec
from llm2vec.loss.utils import load_loss
from llm2vec_da.training import SimCSEDefaultCollator, SimCSETrainer, StopTrainingCallback


def parse_args():
    parser = argparse.ArgumentParser(description='SimCSE Training Script')
    parser.add_argument(
        '--config', 
        type=str, 
        default="configs/simcse/MetaLlama3-swe-dk-wiki-scandi.json",
        help='Path to the configuration JSON file'
    )
    return parser.parse_args()


def main():
    # Parse command line arguments
    args = parse_args()
    
    # Parse arguments from config file
    simcse_parser = HfArgumentParser(
        (SimCSEModelArguments, SimCSEDataTrainingArguments, TrainingArguments, SimCSECustomArguments)
    )
    
    model_args, data_args, training_args, custom_args = simcse_parser.parse_json_file(args.config)
    
    # Setup accelerator and seed
    accelerator = Accelerator()
    transformers.set_seed(training_args.seed)
    
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}
    
    # Load datasets
    datasets = _load_from_files(data_args, model_args)
    train_dataset = PairedDataset(datasets['train'])
    valid_dataset = PairedDataset(datasets['validation'])
    
    # Load examples
    train_examples = [
        train_dataset[i]
        for i in tqdm(
            range(len(train_dataset)),
            desc="Loading train examples...",
            disable=not accelerator.is_main_process
        )
    ]
    
    validation_examples = [
        valid_dataset[i]
        for i in tqdm(
            range(len(valid_dataset)),
            desc="Loading validation examples...",
            disable=not accelerator.is_main_process
        )
    ]
    
    # Initialize model
    model = LLM2Vec.from_pretrained(
        base_model_name_or_path=model_args.model_name_or_path,
        enable_bidirectional=model_args.bidirectional,
        peft_model_name_or_path=model_args.peft_model_name_or_path,
        merge_peft=True,
        pooling_mode=model_args.pooling_mode,
        max_length=model_args.max_seq_length,
        torch_dtype=getattr(torch, model_args.torch_dtype),
        attn_implementation=model_args.attn_implementation,
        attention_dropout=custom_args.simcse_dropout,
        load_in_8bit=True
    )
    
    # Setup training
    train_loss = load_loss(custom_args.loss_class, scale=custom_args.loss_scale)
    data_collator = SimCSEDefaultCollator(model.tokenize)
    
    # Initialize trainer
    trainer = SimCSETrainer(
        model=model,
        args=training_args,
        train_dataset=train_examples,
        eval_dataset=validation_examples,
        data_collator=data_collator,
        tokenizer=model.tokenizer,
        loss_function=train_loss,
    )
    
    # Add callbacks if needed
    if custom_args.stop_after_n_steps is not None:
        trainer.add_callback(StopTrainingCallback(custom_args.stop_after_n_steps))
    trainer.callback_handler.remove_callback(transformers.integrations.integration_utils.WandbCallback)
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main() 