from .model_modifications.llm2vec_class import LLM2Vec
from .model import initialize_peft, get_model_class
from .training import DataCollatorForLanguageModelingWithFullMasking

class llm2vec_da:

    @staticmethod
    def from_pretrained(model_name_or_path="AI-Sweden-Models/Llama-3-8B-instruct", **kwargs):
        return LLM2Vec.from_pretrained(model_name_or_path, **kwargs)