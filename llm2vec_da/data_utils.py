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