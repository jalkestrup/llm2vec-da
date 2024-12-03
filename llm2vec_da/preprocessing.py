from itertools import chain

from datasets.formatting.formatting import LazyBatch
from transformers import PreTrainedTokenizer

def _tokenize_by_line(raw_datasets, tokenizer, data_args, training_args, text_column_name="text",max_seq_length=1024):
    # When using line_by_line, we just tokenize each nonempty line.
    padding = "max_length" if data_args.pad_to_max_length else False

    def tokenize_function(examples):
        # Remove empty lines
        examples[text_column_name] = [
            line
            for line in examples[text_column_name]
            if len(line) > 0 and not line.isspace()
        ]
        return tokenizer(
            examples[text_column_name],
            padding=padding,
            truncation=True,
            max_length=max_seq_length,
            # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
            # receives the `special_tokens_mask`.
            return_special_tokens_mask=True,
        )

    with training_args.main_process_first(desc="dataset map tokenization"):
        if not data_args.streaming:
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=[text_column_name],
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset line_by_line",
            )
        else:
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                remove_columns=[text_column_name],
            )
    return tokenized_datasets


def _tokenize_dataset(raw_datasets, tokenizer, data_args, training_args, text_column_name="text"):
    # Otherwise, we tokenize every text, then concatenate them together before splitting them in smaller parts.
    # We use `return_special_tokens_mask=True` because DataCollatorForLanguageModeling (see below) is more
    # efficient when it receives the `special_tokens_mask`.

    column_names = list(raw_datasets["train"].features)
    assert text_column_name in column_names, f"Provided text_column_name {text_column_name} not in dataset"
    def tokenize_function(examples):
        return tokenizer(
            examples[text_column_name], return_special_tokens_mask=True
        )

    with training_args.main_process_first(desc="dataset map tokenization"):
        if not data_args.streaming:
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on every text in dataset",
            )
        else:
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                remove_columns=column_names,
            )
    return tokenized_datasets

def tokenize_datasets(raw_datasets, tokenizer, data_args, training_args, text_column_name="text", max_seq_length=1024):

    if data_args.line_by_line:
        return _tokenize_by_line(raw_datasets, tokenizer, data_args, training_args, text_column_name=text_column_name,max_seq_length=max_seq_length)
    else:
        return _tokenize_dataset(raw_datasets, tokenizer, data_args, training_args, text_column_name=text_column_name)

# Main data processing function that will concatenate all texts from our dataset and generate chunks of
# max_seq_length.
def group_texts(examples:LazyBatch, max_seq_length=1024):
    """
    This function will receive a batch of texts and return a list of chunks of texts that have length max_seq_length.
    Intended usage with `datasets.map(ds, batched=True)`

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a
    # remainder for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value
    # might be slower to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/process#map
    """
    # Concatenate all texts.
    concatenated_examples = {
        k: list(chain(*examples[k])) for k in examples.keys()
    }
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, and if the total_length < max_seq_length  we exclude this batch and return an empty dict.
    # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
    total_length = (total_length // max_seq_length) * max_seq_length
    # Split by chunks of max_len.
    result = {
        k: [
            t[i : i + max_seq_length]
            for i in range(0, total_length, max_seq_length)
        ]
        for k, t in concatenated_examples.items()
    }
    return result



