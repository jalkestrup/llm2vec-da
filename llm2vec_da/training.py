import logging
import os
from dataclasses import dataclass
from typing import Any, Optional, Tuple, List, Callable
from typing import Dict, Union

import torch
import torch.nn as nn
from llm2vec import LLM2Vec
from llm2vec.dataset.dataset import TrainSample
from transformers import DataCollatorForLanguageModeling, TrainerCallback, Trainer
from transformers.trainer_utils import seed_worker
from torch.utils.data import DataLoader, SequentialSampler

from transformers import (
    LlamaConfig,
    MistralConfig,
    GemmaConfig,
    Qwen2Config,
)

logger = logging.getLogger(__name__)

class DataCollatorForLanguageModelingWithFullMasking(DataCollatorForLanguageModeling):
    def torch_mask_tokens(
        self,
        inputs: Any,
        special_tokens_mask: Optional[Any] = None,
    ) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 100% MASK, 0% random, 0% original.
        """

        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(
                    val, already_has_special_tokens=True
                )
                for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 100% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        inputs[masked_indices] = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.mask_token
        )

        return inputs, labels


class StopTrainingCallback(TrainerCallback):
    def __init__(self, stop_after_n_steps: int):
        self.stop_after_n_steps = stop_after_n_steps

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step >= self.stop_after_n_steps:
            control.should_training_stop = True


class MNTPTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_names = ["labels"]

    def _remove_unused_columns(
        self, dataset: "datasets.Dataset", description: Optional[str] = None
    ):
        return dataset

    # We need a custom save function as we have to save the inner model
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        # model organization is MODEL_TYPEBiForMNTP.model -> MODEL_TYPELBiModel, we have to save the inner model, handled by save_peft_model function of the outer model
        self.model.save_peft_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))


class SimCSETrainer(Trainer):
    def __init__(
        self,
        *args,
        loss_function=None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.loss_function = loss_function

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        features, labels = inputs
        q_reps = self.model(features[0])
        d_reps = self.model(features[1])

        d_reps_neg = None
        if len(features) > 2:
            d_reps_neg = self.model(features[2])

        loss = self.loss_function(q_reps, d_reps, d_reps_neg)

        if return_outputs:
            output = torch.cat(
                [model(row)["sentence_embedding"][:, None] for row in features], dim=1
            )
            return loss, output

        return loss

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        self.model.save(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))


def prepare_for_tokenization(model, text, pooling_mode="mean"):
    if model.config._name_or_path == "meta-llama/Meta-Llama-3-8B-Instruct":
        text = (
            "<|start_header_id|>user<|end_header_id|>\n\n" + text.strip() + "<|eot_id|>"
        )
        return text
    if model.config._name_or_path in [
        "mistralai/Mistral-7B-Instruct-v0.2",
        "meta-llama/Llama-2-7b-chat-hf",
    ]:
        text = "[INST] " + text.strip() + " [/INST]"
    if model.config._name_or_path in [
        "google/gemma-2-9b-it",
    ]:
        text = "<bos><start_of_turn>user\n" + text.strip() + "<end_of_turn>"
    if model.config._name_or_path in [
        "Qwen/Qwen2-1.5B-Instruct",
        "Qwen/Qwen2-7B-Instruct",
    ]:
        text = "<|im_start|>user\n" + text.strip() + "<|im_end|>"
    if pooling_mode == "eos_token":
        if model.config._name_or_path == "meta-llama/Meta-Llama-3-8B":
            text = text.strip() + "<|end_of_text|>"
        elif isinstance(model.config, LlamaConfig) or isinstance(
            model.config, MistralConfig
        ):
            text = text.strip() + " </s>"
        elif isinstance(model.config, GemmaConfig):
            text = text.strip() + "<eos>"
        elif isinstance(model.config, Qwen2Config):
            text = text.strip() + "<|endoftext|>"
    return text


class MixedNegCollator:
    #def __init__(self, model: LLM2Vec):
    def __init__(self, model):
        self.model = model

    def _prep(self, txt):
        return prepare_for_tokenization(self.model, txt,
                                        pooling_mode=self.model.pooling_mode)

    def __call__(self, batch):
        if len(batch) == 0:
            print("⚠️  Empty eval batch encountered")
            return None

        q_texts, p_texts, n_texts, labels = [], [], [], []

        for ex in batch:
            q_texts.append(self._prep(ex.texts[0]))
            p_texts.append(self._prep(ex.texts[1]))

            if len(ex.texts) > 2 and ex.texts[2]:
                n_texts.append(self._prep(ex.texts[2]))

            labels.append(ex.label)

        sent_feat_q = self.model.tokenize(q_texts)          # size B
        sent_feat_p = self.model.tokenize(p_texts)          # size B
        sent_feat_n = (
            self.model.tokenize(n_texts) if n_texts else None
        )      
                                                     # size ≤ B or None

        return {"q":  sent_feat_q,
                "p":  sent_feat_p,
                "n":  sent_feat_n,      # can be None
                "labels": labels,
            }

class SimCSEDefaultCollator:
    def __init__(self, tokenizer:Callable):
        self.tokenizer = tokenizer

    def __call__(self, features: List[TrainSample]):
        batch = features
        num_texts = len(batch[0].texts)
        texts = [[] for _ in range(num_texts)]
        labels = []

        # stack the texts vertically

        for example in batch:
            for idx, text in enumerate(example.texts):
                texts[idx].append(text)
            labels.append(example.label)
        labels = torch.tensor(labels)

        sentence_features = []
        for idx in range(num_texts):
            tokenized = self.tokenizer(texts[idx])
            sentence_features.append(tokenized)

        return sentence_features, labels

class SupervisedDefaultCollator:
    model: LLM2Vec

    def __init__(self, model: LLM2Vec) -> None:
        self.model = model

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch = features
        num_texts = len(batch[0].texts)
        texts = [[] for _ in range(num_texts)]
        labels = []

        for example in batch:
            for idx, text in enumerate(example.texts):
                text = prepare_for_tokenization(
                    self.model, text, pooling_mode=self.model.pooling_mode
                )
                texts[idx].append(text)
            labels.append(example.label)
        labels = torch.tensor(labels)

        sentence_features = []
        for idx in range(num_texts):
            tokenized = self.model.tokenize(texts[idx])
            sentence_features.append(tokenized)

        return sentence_features, labels


class SupervisedTrainer(Trainer):
    def __init__(
        self,
        *args,
        loss_function=None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.loss_function = loss_function

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        q_rep = model(inputs["q"])
        p_rep = model(inputs["p"])
        n_rep = model(inputs["n"]) if inputs["n"] is not None else None

        loss = self.loss_function(q_rep, p_rep, n_rep)

        if return_outputs:
            # optional extra output, keep if you need it for something
            output = torch.cat(
                [q_rep["sentence_embedding"][:, None],
                p_rep["sentence_embedding"][:, None],
                n_rep["sentence_embedding"][:, None] if n_rep is not None else
                torch.empty_like(q_rep["sentence_embedding"][:, None])],
                dim=1
            )
            return loss, output

        return loss


    def get_train_dataloader(self) -> DataLoader:
        # Copying most of the code from the parent class, changing the sampler to SequentialSampler
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator

        data_collator = self._get_collator_with_removed_columns(
            data_collator, description="training"
        )

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "drop_last": self.args.dataloader_drop_last,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            # Changing from random sampler to sequential sampler
            dataloader_params["sampler"] = SequentialSampler(train_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

    def get_eval_dataloader(self, eval_dataset: Optional[Any] = None) -> DataLoader:
        dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        data_collator = self._get_collator_with_removed_columns(
            self.data_collator, description="evaluation"
        )

        dataloader_params = {
            "batch_size":      self.args.per_device_eval_batch_size,
            "collate_fn":      data_collator,
            "sampler":         SequentialSampler(dataset),
            "drop_last":       False, 
            "num_workers":     self.args.dataloader_num_workers,
            "pin_memory":      self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }
        return self.accelerator.prepare(DataLoader(dataset, **dataloader_params))

    # Overwrite the function used for eval to use the correct loss function
    def prediction_step(
        self, model, inputs, prediction_loss_only=True, ignore_keys=None
    ):
        if inputs is None:                 # skip empty batches
            return None, None, None

        loss = self.compute_loss(model, inputs, return_outputs=False)
        return loss.detach(), None, None   # logits / labels not needed

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        self.model.save(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
