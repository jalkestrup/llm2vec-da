# LLM2Vec-DA: Training Code for Scandinavian Language Embeddings

This repository contains the code and training scripts for training embedding models for Scandinavian languages using the LLM2Vec approach. The code has been used to train models that achieve state-of-the-art performance on the [Scandinavian Embedding Benchmark (SEB)](https://kennethenevoldsen.github.io/scandinavian-embedding-benchmark/).

## Notable Models Trained with This Code

The code in this repository has been used to train several models, including:

1. **TTC-L2V-supervised-2** ([Hugging Face](https://huggingface.co/jealk/TTC-L2V-supervised-2)): The current state-of-the-art supervised embedding model for Danish, Swedish, and Norwegian.
2. **TTC-L2V-unsupervised-1** ([Hugging Face](https://huggingface.co/jealk/TTC-L2V-unsupervised-1)): The best performing unsupervised embedding model for Danish text.

## Repository Structure

- `llm2vec_da/`: Core library code, a rewrite of the [LLM2Vec](https://github.com/McGill-NLP/llm2vec) library
- `configs/`: Training configurations for different models
  - `mntp/`: Configurations for MNTP (Masked Next Token Prediction) training
  - `supervised/`: Configurations for supervised training
  - `simcse/`: Configurations for SimCSE training
- `seb/`: Code for running evaluations on the Scandinavian Embedding Benchmark
- Training scripts:
  - `1_mntp_data_tokenize.ipynb`: Notebook for preparing MNTP training data
  - `2_mntp_training.py`: MNTP training script
  - `3_simcse_training.py`: SimCSE training script
  - `4_supervised_training.py`: Supervised training script

## Training Pipeline

The models are trained in a multi-stage process:

1. **MNTP Training**: Initial training using Masked Next Token Prediction
   - First run `1_mntp_data_tokenize.ipynb` to prepare the training data
   - Then use `2_mntp_training.py` for the actual training
   - Configuration in `configs/mntp/`

2. **Supervised Training**: Fine-tuning on supervised data
   - Use `4_supervised_training.py`
   - Configuration in `configs/supervised/`
   - This produces the state-of-the-art TTC-L2V-supervised-2 model

3. **SimCSE Training** (Optional): Unsupervised training using SimCSE
   - Use `3_simcse_training.py`
   - Configuration in `configs/simcse/`
   - This produces the TTC-L2V-unsupervised-1 model

## Evaluation

The `seb/` directory contains code for evaluating models on the Scandinavian Embedding Benchmark. This includes:
- Running evaluations
- Visualizing results
- Comparing model performance

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- Accelerate
- Other dependencies listed in `requirements.txt`

## License

This project is licensed under the MIT License - see the LICENSE file for details.




