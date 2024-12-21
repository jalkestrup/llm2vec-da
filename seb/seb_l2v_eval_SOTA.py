# Import necessary libraries
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
from peft import PeftModel

from llm2vec import LLM2Vec
from typing import Any, Optional, TypeVar
from collections.abc import Iterable, Sequence
from transformers import AutoTokenizer, BatchEncoding
from tqdm import tqdm
from itertools import islice
import numpy as np
import seb
from seb.interfaces.task import Task
import pickle
import os
from dotenv import load_dotenv
from huggingface_hub import login


T = TypeVar("T")

# ----------- Authenticating huggingface -----------

load_dotenv()

hf_token = os.getenv('HF_TOKEN')

if hf_token:
    login(token=hf_token)

else:
    raise ValueError("Hugging Face token not found. Please ensure it's set in the .env file.")


# ----------- Define LLM models ------------

hf_mntp_model = 'jealk/llm2vec-da-mntp'
hf_simcse_model = 'jealk/TTC-unsupervised-1'
hf_simcse_revision = ""

# Model name to save in SEB and local pkl {original model name}-{model type [instruct, llm2vec]}-{ft type}-{ft dataset}-{ft steps}
seb_model_name = 'llama-8b-swe-llm2vec-mntp-dkwiki-simcse-scandiwiki-1000-steps'

# ----------- Loading the llm2vec model according to repo -----------

# Load base mistral model with custom code
tokenizer = AutoTokenizer.from_pretrained(hf_mntp_model)
config = AutoConfig.from_pretrained(hf_mntp_model, trust_remote_code=True)

model = AutoModel.from_pretrained(
    hf_mntp_model,
    config=config,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="cuda" if torch.cuda.is_available() else "cpu",
)

# Load MNTP LoRA weights
model = PeftModel.from_pretrained(model, hf_mntp_model)

# Merge and unload
model = model.merge_and_unload()

#Check if hf_simcse_revision is defined in this script
if hf_simcse_revision:
    model = PeftModel.from_pretrained(model, hf_simcse_model, revision=hf_simcse_revision)
else:
    model = PeftModel.from_pretrained(model, hf_simcse_model)

# Create a wrapper instance for encoding and pooling operations
l2v = LLM2Vec(model, tokenizer, pooling_mode="mean", max_length=512)

# ----------- Benchmark Setup -----------

model_name = hf_simcse_model

def task_to_instruction(task: Task) -> str:
    if task.task_type in ["STS"]:
        return "Retrieve semantically similar text"
    if task.task_type in ["Summarization"]:
        return "Given a news summary, retrieve other semantically similar summaries"
    if task.task_type in ["BitextMining"]:
        task_name_to_instruct: dict[str, str] = {
            "Bornholm Parallel": "Retrieve parallel sentences in Danish and Bornholmsk",
            "Norwegian courts": "Retrieve parallel sentences in Norwegian Bokmål and Nynorsk",
        }
        default_instruction = "Retrieve parallel sentences."
        return task_name_to_instruct.get(task.name, default_instruction)
    if task.task_type in ["Classification"]:
        task_name_to_instruct: dict[str, str] = {
            "Angry Tweets": "Classify Danish tweets by sentiment. (positive, negative, neutral)",
            "DKHate": "Classify Danish tweets based on offensiveness (offensive, not offensive)",
            "Da Political Comments": "Classify Danish political comments for sentiment",
            "DaLAJ": "Classify texts based on linguistic acceptability in Swedish",
            "LCC": "Classify texts based on sentiment",
            "Language Identification": "Classify texts based on language",
            "Massive Intent": "Given a user utterance as query, find the user intents",
            "Massive Scenario": "Given a user utterance as query, find the user scenarios",
            "NoReC": "Classify Norwegian reviews by sentiment",
            "SweReC": "Classify Swedish reviews by sentiment",
            "Norwegian parliament": "Classify parliament speeches in Norwegian based on political affiliation",
            "ScaLA": "Classify passages in Scandinavian Languages based on linguistic acceptability",
        }
        default_instruction = "Classify user passages"
        return task_name_to_instruct.get(task.name, default_instruction)
    if task.task_type in ["Clustering"]:
        task_name_to_instruct: dict[str, str] = {
            "ArxivClusteringP2P": "Identify the main and secondary category of Arxiv papers based on the titles and abstracts",
            "VG Clustering": "Identify the categories (e.g. sports) of given articles in Norwegian",
            "SNL Clustering": "Identify categories in a Norwegian lexicon",
            "SwednClustering": "Identify news categories in Swedish passages",
        }
        default_instruction = "Identify categories in user passages"
        return task_name_to_instruct.get(task.name, default_instruction)
    if task.task_type in ["Reranking"]:
        return "Retrieve semantically similar passages."
    if task.task_type in ["Retrieval"]:
        task_name_to_instruct: dict[str, str] = {
            "Twitterhjerne": "Retrieve answers to questions asked in Danish tweets",
            "SwednRetrieval": "Given a Swedish news headline retrieve summaries or news articles",
            "TV2Nord Retrieval": "Given a summary of a Danish news article retrieve the corresponding news article",
            "DanFEVER": "Given a claim in Danish, retrieve documents that support the claim",
            "SNL Retrieval": "Given a lexicon headline in Norwegian, retrieve its article",
            "NorQuad": "Given a question in Norwegian, retrieve the answer from Wikipedia articles",
            "SweFAQ": "Retrieve answers given questions in Swedish",
            "ArguAna": "Given a claim, find documents that refute the claim",
            "ClimateFEVER": "Given a claim about climate change, retrieve documents that support or refute the claim",
        }
        default_instruction = "Retrieve text based on user query."
        return task_name_to_instruct.get(task.name, default_instruction)
    return ""


def batched(iterable: Iterable[T], n: int) -> Iterable[tuple[T, ...]]:
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch

# Custom SEB Encoder Model
class SEB_L2V(seb.Encoder):
    def __init__(self, model_name: str, max_length: int, max_batch_size: Optional[int] = None, **kwargs: Any):
        self.model = l2v
        self.max_length = max_length
        self.max_batch_size = max_batch_size
        
    def preprocess(self, sentences: Sequence[str], instruction: str) -> Any:
        if instruction is not None:
            sentences = [[instruction, sentence] for sentence in sentences]
        return sentences
    
    def encode(
        self,
        sentences: list[str],
        *,
        task: seb.Task,
        batch_size: int = 128,
        **kwargs: Any,
    ) -> np.ndarray:

        if self.max_batch_size and batch_size > self.max_batch_size:
            batch_size = self.max_batch_size

        batched_embeddings = []
        if task is not None:
            instruction = task_to_instruction(task)
        else:
            instruction = None

        for batch in tqdm(batched(sentences, batch_size)):
            preprocessed_batch = self.preprocess(batch, instruction=instruction)
            with torch.inference_mode():
                embedded_batch = self.model.encode(preprocessed_batch)
            batched_embeddings.append(embedded_batch)

        return torch.cat(batched_embeddings).numpy()


@seb.models.register(seb_model_name)
def create_my_model() -> seb.SebModel:
    meta = seb.ModelMeta(
        name=seb_model_name,
        huggingface_name=hf_simcse_model,
        reference=f"https://huggingface.co/{hf_simcse_model}",
        languages=['da'],
        embedding_size=4096,
    )
    return seb.SebModel(
        encoder=SEB_L2V(l2v, max_length=256, max_batch_size=128),
        meta=meta,
    )

# ----------- Running the Benchmark -----------

def run_benchmark():
    models = [seb.get_model(seb_model_name)]
    benchmark = seb.Benchmark(['da'])
    results = benchmark.evaluate_models(models=models, use_cache=False)
    #Save pickle
    with open(f"SEB_eval_{seb_model_name}.pkl", 'wb') as f:
        pickle.dump(results, f)
    avg_score = np.mean([res.get_main_score() for res in results])
    print(f'\nAverage results: {avg_score}')
    print(f'\nFull Results:\n{results}')

if __name__ == "__main__":
    run_benchmark()