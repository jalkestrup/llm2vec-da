# Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
# or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
# (the dataset will be downloaded automatically from the datasets Hub
#
# For CSV/JSON files, this script will use the column called 'text' or the first column. You can easily tweak this
# behavior (see below)
#
# In distributed training, the load_dataset function guarantee that only one local process can concurrently
# download the dataset.
from datasets import load_dataset
from dataclasses import dataclass
from typing import Union, List
import torch
from tqdm import tqdm


def _load_from_hub(data_args, model_args):
    raw_datasets = load_dataset(
        data_args.dataset_name,
        data_args.dataset_config_name,
        cache_dir=model_args.cache_dir,
        token=model_args.token,
        streaming=data_args.streaming,
    )
    if "validation" not in raw_datasets.keys():
        raw_datasets["validation"] = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            split=f"train[:{data_args.validation_split_percentage}%]",
            cache_dir=model_args.cache_dir,
            token=model_args.token,
            streaming=data_args.streaming,
        )
        raw_datasets["train"] = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            split=f"train[{data_args.validation_split_percentage}%:]",
            cache_dir=model_args.cache_dir,
            token=model_args.token,
            streaming=data_args.streaming,
        )
    return raw_datasets


def load_raw_datasets(data_args, model_args):
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        return _load_from_hub(data_args, model_args)
    else:
        raw_datasets = _load_from_files(data_args, model_args)
    return raw_datasets


def _load_from_files(data_args, model_args):
    data_files = {}
    if data_args.train_file is not None:
        data_files["train"] = data_args.train_file
        extension = data_args.train_file.split(".")[-1]
    if data_args.validation_file is not None:
        data_files["validation"] = data_args.validation_file
        extension = data_args.validation_file.split(".")[-1]
    if extension == "txt":
        extension = "text"
    raw_datasets = load_dataset(
        extension,
        data_files=data_files,
        cache_dir=model_args.cache_dir,
        token=model_args.token,
    )
    # If no validation data is there, validation_split_percentage will be used to divide the dataset.
    if "validation" not in raw_datasets.keys():
        raw_datasets["validation"] = load_dataset(
            extension,
            data_files=data_files,
            split=f"train[:{data_args.validation_split_percentage}%]",
            cache_dir=model_args.cache_dir,
            token=model_args.token,
        )
        raw_datasets["train"] = load_dataset(
            extension,
            data_files=data_files,
            split=f"train[{data_args.validation_split_percentage}%:]",
            cache_dir=model_args.cache_dir,
            token=model_args.token,
        )
    return raw_datasets


@dataclass
class DataSample:
    id_: int
    query: str
    positive: str
    negative: str = None
    task_name: str = None


class TrainSample:
    """
    Structure for one input example with texts, the label and a unique id
    """

    def __init__(
        self, guid: str = "", texts: List[str] = None, label: Union[int, float] = 0
    ):
        """
        Creates one TrainSample with the given texts, guid and label


        :param guid
            id for the example
        :param texts
            the texts for the example.
        :param label
            the label for the example
        """
        self.guid = guid
        self.texts = texts
        self.label = label

    def __str__(self):
        return "<TrainSample> label: {}, texts: {}".format(
            str(self.label), "; ".join(self.texts)
        )


class Dataset(torch.utils.data.Dataset):
    """
    Abstract class for datasets
    """
    
    def load_data(self, file_path: str = None):
        raise NotImplementedError()

    def __getitem__(self, index):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()



class PairedDataset(Dataset):
    """
    Prepares a dataset for SIMCSE training by pairing the same text with itself as positive example
    """

    def __init__(self, data:Dataset):
        self.data = []
        for i, t in enumerate(data['text']):
            self.data.append(DataSample(id_=i, query=t, positive=t))

    def load_data(self, file_path: str = None):
        pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        return TrainSample(texts=[sample.query, sample.positive], label=1.0)

import json
import random
import os

from accelerate.logging import get_logger

logger = get_logger(__name__, log_level="INFO")

E5_EMBEDDING_PROMPTS = {
    "allnli": [
        "Given a premise, retrieve a hypothesis that is entailed by the premise",
        "Retrieve semantically similar text",
    ],
    "dureader": "Given a Chinese search query, retrieve web passages that answer the question",
    "eli5_question_answer": "Provided a user question, retrieve the highest voted answers on Reddit ELI5 forum",
    "fever": "Given a claim, retrieve documents that support or refute the claim",
    "hotpot_qa": "Given a multi-hop question, retrieve documents that can help answer the question",
    "miracl": "Given a question, retrieve Wikipedia passages that answer the question",
    "mrtydi": "Given a question, retrieve Wikipedia passages that answer the question",
    "msmarco_passage": "Given a web search query, retrieve relevant passages that answer the query",
    "msmarco_document": "Given a web search query, retrieve relevant documents that answer the query",
    "nq": "Given a question, retrieve Wikipedia passages that answer the question",
    "quora_duplicates": [
        "Given a question, retrieve questions that are semantically equivalent to the given question",
        "Find questions that have the same meaning as the input question",
    ],
    "squad": "Retrieve Wikipedia passages that answer the question",
    "t2ranking": "Given a Chinese search query, retrieve web passages that answer the question",
    "trivia_qa": "Retrieve Wikipedia passages that answer the question",
}

EMBEDDING_PROMPTS = {
    "retrieval-dk": "Givet et spørgsmål, find den tekst der bedst besvarer spørgsmålet",
    "classification-dk": "Klassificér den følgende tekst",
    "unit-triple-dk": "Givet en tekst, rangér de andre efter semantisk relevans",
    "text-matching-long-dk": "Givet en lang tekst, find en anden tekst med tilsvarende indhold og betydning",
    "text-matching-short-dk": "Givet en kort tekst, find en anden tekst med tilsvarende indhold og betydning",
    "retrieval-no": "Gitt et spørsmål, finn teksten som best svarer på spørsmålet",
    "classification-no": "Klassifiser følgende tekst",
    "unit-triple-no": "Gitt en tekst, ranger de andre etter semantisk relevans",
    "text-matching-long-no": "Gitt en lang tekst, finn en annen tekst med tilsvarende innhold og betydning",
    "text-matching-short-no": "Gitt en kort tekst, finn en annen tekst med tilsvarende innhold og betydning",
    "retrieval-se": "Givet en fråga, hitta den text som bäst besvarar frågan",
    "classification-se": "Klassificera följande text",
    "unit-triple-se": "Givet en text, rangordna de andra efter semantisk relevans",
    "text-matching-long-se": "Givet en lång text, hitta en annan text med motsvarande innehåll och betydelse",
    "text-matching-short-se": "Givet en kort text, hitta en annan text med motsvarande innehåll och betydelse",
}


class E5Data(Dataset):
    def __init__(
        self,
        dataset_name: str = "E5",
        split: str = "validation",
        file_path: str = "cache/echo-data",
        effective_batch_size: int = 32,
        shuffle_individual_datasets: bool = True,
        separator: str = "!@#$%^&*()",
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.effective_batch_size = effective_batch_size
        self.shuffle_individual_datasets = shuffle_individual_datasets
        self.separator = separator

        self.data = []
        self.load_data(file_path)

    def __len__(self):
        return len(self.data)

    def load_data(self, file_path: str = None):
        logger.info(f"Loading E5 data from {file_path}...")
        # file path is actually a directory

        data_map = {}
        all_samples = []
        id_ = 0
        for dataset in E5_EMBEDDING_PROMPTS:
            logger.info(f"Loading dataset {dataset}...")
            if dataset not in data_map:
                data_map[dataset] = []
            with open(os.path.join(file_path, f"{dataset}.jsonl"), "r") as f:
                dataset_samples = f.readlines()

            dataset_samples = [json.loads(d) for d in dataset_samples]

            for i, sample in enumerate(dataset_samples):
                instruction = (
                    E5_EMBEDDING_PROMPTS[dataset]
                    if isinstance(E5_EMBEDDING_PROMPTS[dataset], str)
                    else E5_EMBEDDING_PROMPTS[dataset][i % 2]
                )
                query = f"{instruction}; " + self.separator + sample["query"]
                if dataset in [
                    "allnli_split2",
                    "quora_duplicates_split1",
                    "quora_duplicates_split2",
                ]:
                    pos = (
                        f"{E5_EMBEDDING_PROMPTS[dataset]}; "
                        + self.separator
                        + sample["positive"]
                    )
                    neg = (
                        f"{E5_EMBEDDING_PROMPTS[dataset]}; "
                        + self.separator
                        + sample["negative"]
                    )
                else:
                    pos = self.separator + sample["positive"]
                    neg = self.separator + sample["negative"]

                data_map[dataset].append(id_)

                all_samples.append(
                    DataSample(
                        id_=id_,
                        query=query,
                        positive=pos,
                        negative=neg,
                        task_name=dataset,
                    )
                )
                id_ += 1

        # combine split1 and split2
        new_data_map = {}
        for dataset in data_map:
            new_dataset = dataset.replace("_split1", "").replace("_split2", "")
            if new_dataset not in new_data_map:
                new_data_map[new_dataset] = []
            new_data_map[new_dataset] += data_map[dataset]
        data_map = new_data_map

        if self.shuffle_individual_datasets:
            for task, samples in data_map.items():
                random.shuffle(samples)

        datasets = list(data_map.keys())

        logger.info(
            f"Batching Echo data properly for effective batch size of {self.effective_batch_size}..."
        )
        all_batches = []
        for dataset in datasets:
            dataset_samples = data_map[dataset]
            for i in range(0, len(dataset_samples), self.effective_batch_size):
                batch = dataset_samples[i : i + self.effective_batch_size]
                if len(batch) == self.effective_batch_size:
                    all_batches.append(batch)
                else:
                    logger.info(f"Skip 1 batch for dataset {dataset}.")
        random.shuffle(all_batches)

        final_idx_order = []
        for batch in all_batches:
            for idx in batch:
                final_idx_order.append(idx)

        self.data = [all_samples[idx] for idx in final_idx_order]
        logger.info(f"Loaded {len(self.data)} samples.")

    def __getitem__(self, index):
        sample = self.data[index]
        if self.split == "train":
            return TrainSample(
                texts=[sample.query, sample.positive, sample.negative], label=1.0
            )
        elif self.split == "validation":
            assert False, "E5Data does not have a validation split."



class NordicE5Data(Dataset):
    """
    A dataset class for loading and processing data from a Hugging Face dataset to a datasample following E5 datastructure.
    
    This class handles loading instruction-based samples with queries, positive examples,
    and optional negative examples. It processes the data into batches suitable for training.

    Args:
        hf_dataset: The dataset to load from (local or remote)
        [Optional] instruction_column (str): Column name for instructions. Defaults to 'instruction'. Prepends the instruction to the query.
        query_column (str): Column name for queries. Defaults to 'query'
        pos_column (str): Column name for positive examples. Defaults to 'positive'
        [Optional] neg_column (str): Column name for negative examples. Defaults to 'negative'
        [Optional] task_column (str): Column name for task labels. Task is used to group the data by task during batching.
        split (str): Dataset split to use. Defaults to "train"
        effective_batch_size (int): Size of batches to create, accounting for parallel processes.
        separator (str): Separator string between text segments. Defaults to "!@#$%^&*()"
    """

    def __init__(
        self,
        hf_dataset,
        instruction_column = 'instruction',
        query_column = 'query',
        pos_column = 'positive',
        neg_column = 'negative',
        task_column = 'task',
        split: str = "train",
        effective_batch_size: int = 32,
        separator: str = "!@#$%^&*()", #Note default of LLM2Vec is !@#$%^&*() , changing this would also have to be changed in the llm2vec lib when encoding/decoding
    ):
        self.instruction_column = instruction_column
        self.query_column = query_column
        self.pos_column = pos_column
        self.neg_column = neg_column
        self.task_column = task_column
        self.split = split
        self.effective_batch_size = effective_batch_size
        self.separator = separator
        self.data = []
        self.load_data(hf_dataset)

    def __len__(self):
        return len(self.data)

    def load_data(self, hf_dataset):
        # 1) Convert the hf dataset to a list of DataSamples
        all_samples = []
        for idx, row in tqdm(enumerate(hf_dataset), total=len(hf_dataset), desc='Loading dataset'):
            
            # If no query and positive example, skip the example
            if self.query_column not in row or self.pos_column not in row:
                logger.warning(f"No query or positive example found for example {idx}, skipping")
                continue

            # If instruction column is provided, prepend the instruction to the query
            if self.instruction_column:
                instruction = row[self.instruction_column]
                query =  f"{instruction}; {self.separator}{row[self.query_column]}"
            else:
                query =  f"{row[self.query_column]}"
        
            pos   =  f"{self.separator}{row[self.pos_column]}"

            # If negative column is provided include negative example
            neg_raw = row[self.neg_column]
            if neg_raw is None or neg_raw.strip().lower() in {"", "none", "null"}:
                neg = None
            else:
                neg   =  f"{self.separator}{row[self.neg_column]}"

            # If task column is provided include task name as to group batches per task
            if row[self.task_column]:
                task  =  row[self.task_column]
            else:
                task = None

            all_samples.append(
                DataSample(
                    id_=idx,
                    query=query,
                    positive=pos,
                    negative=neg,
                    task_name=task
                )
            )

        # First, group samples by task
        task_samples = {}
        for idx, sample in tqdm(enumerate(all_samples), total=len(all_samples), desc='Grouping data by task'):
            task = sample.task_name
            if task not in task_samples:
                task_samples[task] = []
            task_samples[task].append(sample)

        logger.info(f"Batching data for effective batch size = {self.effective_batch_size} ...")
        batched_data = []

        # Create full batches for each task
        for task, samples in tqdm(task_samples.items(), total=len(task_samples), desc='Batching data'):
            task_batches = []
            for i in range(0, len(samples), self.effective_batch_size):
                batch = samples[i : i + self.effective_batch_size]
                if len(batch) == self.effective_batch_size:
                    task_batches.append(batch)
                else:
                    logger.info(f"Skipping partial batch of {len(batch)} samples for task {task}")
            
            if task_batches:  # If we got any full batches for this task
                batched_data.extend(task_batches)

        # Shuffle the batches to mix tasks during training
        random.shuffle(batched_data)

        # Flatten while maintaining batch boundaries
        self.data = [sample for batch in batched_data for sample in batch]
        logger.info(f"Loaded and batched {len(self.data)} samples from {len(task_samples)} tasks")

    def __getitem__(self, index):
        sample = self.data[index]
        texts = [sample.query, sample.positive]
        if sample.negative is not None:          
            texts.append(sample.negative)
        return TrainSample(texts=texts, label=1.0)
        
def custom_dataset(hf_dataset,
                      effective_batch_size):
    
    dataset_map = {
        "nordic-embedding-training-data": NordicE5Data
    }

    if hf_dataset.info.dataset_name in dataset_map:
        return dataset_map[hf_dataset.info.dataset_name](hf_dataset,
                                                        effective_batch_size=effective_batch_size)
    else:
        raise ValueError(f"Dataset {hf_dataset.info.dataset_name} not found in dataset_map")
