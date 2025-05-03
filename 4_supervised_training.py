# Standard library
import os
import argparse
from typing import Optional
os.environ["HF_LOG_LEVEL"] = "error"          
os.environ["HF_HUB_LOG_LEVEL"] = "error"      

# Third-party libraries
import transformers
from transformers import HfArgumentParser, TrainingArguments
from tqdm import tqdm
from accelerate import Accelerator, DistributedDataParallelKwargs
from huggingface_hub import HfApi
from datasets import load_dataset
from dotenv import load_dotenv
import torch
from tqdm import tqdm

# Local application/library imports
from llm2vec_da import LLM2Vec
from llm2vec_da.model import initialize_peft
from llm2vec_da.data_utils import custom_dataset
from llm2vec_da.loss.utils import load_loss
from llm2vec_da.training import MixedNegCollator, SupervisedTrainer, StopTrainingCallback
from llm2vec_da.arguments import EmbeddingModelArguments, DataTrainingArguments, CustomArguments

class PEFTSupervisedTrainer(SupervisedTrainer):
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Save only the PEFT adapter
        self.model.save(output_dir, save_mode="peft_only")
        
        # Save tokenizer
        self.tokenizer.save_pretrained(output_dir)
            
        # Save training arguments
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))


def parse_args():
    parser = argparse.ArgumentParser(description='Supervised Training Script')
    parser.add_argument(
        '--config', 
        type=str, 
        default="configs/supervised/MetaLlama3-sheared.json",
        help='Path to the configuration JSON file'
    )
    return parser.parse_args()

def main():

    if '/teamspace' in os.getcwd():
        os.chdir('/teamspace/studios/this_studio/llm2vec-da')
        # Hmm lighting AI studio changed to the below ..?
        #os.chdir('/home/zeus/content/llm2vec-da')

    args = parse_args()
    load_dotenv()

    api = HfApi(token=os.getenv("HF_TOKEN"))

    supervised_parser = HfArgumentParser(
            (EmbeddingModelArguments, DataTrainingArguments, TrainingArguments, CustomArguments)
        )

    model_args, data_args, training_args, custom_args = supervised_parser.parse_json_file(
            args.config
        )

    if training_args.ddp_find_unused_parameters:
        kwargs = [
            DistributedDataParallelKwargs(
                dim=0,
                broadcast_buffers=True,
                bucket_cap_mb=25,
                find_unused_parameters=True,
                check_reduction=False,
                gradient_as_bucket_view=False,
            )
        ]
    else:
        kwargs = []

    accelerator = Accelerator(kwargs_handlers=kwargs)
    transformers.set_seed(training_args.seed)

    #ABSOLUTELY CRITICAL OR WILL CAUSE OBSCURE NO GRAD ERROR THAT TOOK HALF A DAY TO IDENTIFY
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}

    dataset = load_dataset(data_args.dataset_name,
    split="train[:10%]")
    #,columns=['query', 'positive', 'negative', 'instruction', 'task'])

    # Split the dataset into 95% train and 5% test
    split_dataset = dataset.train_test_split(test_size=0.05, seed=42)

    # Define the splits
    train_dataset = split_dataset['train']
    valid_dataset = split_dataset['test']

    train_dataset_e5 = custom_dataset(train_dataset, 
                                    effective_batch_size=training_args.per_device_train_batch_size* accelerator.num_processes)

    valid_dataset_e5 = custom_dataset(valid_dataset,
                                    effective_batch_size=training_args.per_device_train_batch_size* accelerator.num_processes)

    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )

    #training_args.gradient_checkpointing = False   # turn it off
    model = LLM2Vec.from_pretrained(
        base_model_name_or_path=model_args.model_name_or_path,
        enable_bidirectional=model_args.bidirectional,
        peft_model_name_or_path="McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp",
        merge_peft=True,
        pooling_mode=model_args.pooling_mode,
        max_length=data_args.max_seq_length,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=model_args.low_cpu_mem_usage,
        attn_implementation="sdpa", #OBS SET BACK TO FLASH ATTENTION WHEN RUNNING ON A100 GPU!!
    )

    # peft_model = initialize_peft(
    #     model.model,
    #     lora_r=custom_args.lora_r,
    #     lora_alpha=2 * custom_args.lora_r,
    #     lora_dropout=custom_args.lora_dropout,
    # )

    # model.model = peft_model.model
    # model organization is LLM2VecModel.model -> HF Model, we have to apply PEFT to the inner model
 
    
    model.model = initialize_peft(
        model.model,
        lora_r=custom_args.lora_r,
        lora_alpha=2 * custom_args.lora_r,
        lora_dropout=custom_args.lora_dropout,
    )

    tokenizer = model.tokenizer
    data_collator = MixedNegCollator(model) 

    # Load train examples into memory
    train_examples = [
        train_dataset_e5[i]
        for i in tqdm(
            range(len(train_dataset_e5)),
            desc="Loading train examples...",
            disable=not accelerator.is_main_process,
        )
    ]

    valid_examples = [
        valid_dataset_e5[i]
        for i in tqdm(
            range(len(valid_dataset_e5)),
            desc="Loading valid examples...",
            disable=not accelerator.is_main_process,
        )
    ]

    # Setup wandb
    os.environ["WANDB_PROJECT"] = custom_args.wandb_project
    os.environ["WANDB_LOG_MODEL"] = custom_args.wandb_log_model
    if custom_args.wandb_run_group:
        os.environ["WANDB_RUN_GROUP"] = custom_args.wandb_run_group
    if custom_args.wandb_watch:
        os.environ["WANDB_WATCH"] = custom_args.wandb_watch

    os.environ["WANDB_LOG_MODEL"]="false"


    train_loss = load_loss(custom_args.loss_class, scale=custom_args.loss_scale)

    trainer = PEFTSupervisedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_examples,
        eval_dataset=valid_examples,
        data_collator=data_collator,
        tokenizer=tokenizer,
        loss_function=train_loss,
    )

    if custom_args.stop_after_n_steps is not None:
        trainer.add_callback(StopTrainingCallback(custom_args.stop_after_n_steps))

    trainer.train()

if __name__ == "__main__":
    main() 
