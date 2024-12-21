# *LLM-BASED EMBEDDING MODELS IN LOW RESSOURCE LANGUAGES*
##  DTU 02456, Deep Learning

## Repo overview
This repository contains code for the DTU Course 02456, Deep Learning, and has been made to accomodate the need for a clean repository with runnable notebooks.
The repo also contains the poster from the oral exam, placed in root as *'LLM Based Embedding Models in Low Ressource Languages poster.pdf'*.

For practical purposes of running training jobs on a remote GPU, the primary results from the hand-in has been obtained by running a the various training .py files within the /experiments folder of this repo https://github.com/jalkestrup/llm2vec-dtu - which is a forked and slightly modified version of the original LLM2Vec repo https://github.com/McGill-NLP/llm2vec/ (to accomodate for new datasets).

The datasets created during this project is available in my HF profile, and the most succesfull fine-tunes are likewise uploaded as models to the hub: https://huggingface.co/jealk .

## Replicating results on SEB
To replicate the results on the Scandinavian Embedding Benchmark, run the .py file seb_l2v_eval_SOTA.py which has the PEFT models pre-set as variables. If one wish to run SEB eval on another model, simply replace the following arguments and re-run:

hf_mntp_model = 'jealk/llm2vec-da-mntp'
hf_simcse_model = 'jealk/TTC-unsupervised-1'
seb_model_name = 'llama-8b-swe-llm2vec-mntp-dkwiki-simcse-scandiwiki-1000-steps'

### Logs of previous run
All previous evaluations are placed as .pkl and log files in the sub-folder /evals

### Visualisations of training and SEB results
Figures of train loss and SEB results are plotted by running the file llm2vec_train_results.ipynb from end-to-end. 
Training steps are hard-copied from the respective training output folders into this file.

*Note: During the project, actual SEB evals and repo-updates has been done in this repo, https://github.com/jalkestrup/dtu-deep-project , of which the relevant files have been merged into this repo for clarity*

## Training, MTP, SimcSE and supervised
Configs for all fine-tunes are placed in /configs folder from which underlying dataset and model parameters is defined

### Creating datasets
To re-create the Danish dataset used for MNTP and SimCSE training, run the 0_dk_wiki_data_create.ipynb . Do not run the cells that attempt to huggingface repo as these will fail, instead save to local /data. Similarly for the scandinavian and supervised dataset, using 0_scandi_wiki_data_create.ipynb and 0_supervised_data_create.

### Preparing for training
Run 1_mntp_data_tokenize.ipynb to create the tokenized dataset

### MNTP training
**Note: This requires >50GB GPU RAM and was run on a A100, 80 GB GPU**. 1000 steps, training time approximately 2 hours.
Run 2_mntp_training.ipynb

### SimCSE
**Note: This requires >50GB GPU RAM and was run on a A100, 80 GB GPU**. 1000 steps, training time approximately 3 hours.
Run 3_simcse_training.ipynb

### Supervised
**Note: This requires >50GB GPU RAM and was run on a A100, 80 GB GPU**.  500 steps w. grad-accum=4, training time approximately 14 hours.
4_supervised_training is not implemented, refer to the link in the file for the .py file run to create the model refered to in the hand-in.




