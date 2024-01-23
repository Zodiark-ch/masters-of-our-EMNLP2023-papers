# Our Papers
Pytorch code for EMNLP 2023 accepted-main paper "How to Enhance Causal Discrimination of Utterances: A Case on Affective Reasoning" 
and TKDE paper "Learning a Structural Causal Model for Intuition Reasoning in Conversation" (early access) 

the bibtexs are 

@inproceedings{
chen2023how,
title={How to Enhance Causal Discrimination of Utterances: A Case on Affective Reasoning},
author={Chen, Hang and Yang, Xinyu and Luo, Jing and Zhu, Wenjing},
booktitle={The 2023 Conference on Empirical Methods in Natural Language Processing},
year={2023},
url={https://openreview.net/forum?id=x7zquRQfoB}
}

@article{chen2024learning,
  title={Learning a structural causal model for intuition reasoning in conversation},
  author={Chen, Hang and Liao, Bingyu and Luo, Jing and Zhu, Wenjing and Yang, Xinyu},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  year={2024},
  publisher={IEEE}
}

## Requirements
* Python 
* PyTorch 
* Transformers


## Preparation

### Datasets and Utterance Feature
You can download the dataset from https://drive.google.com/file/d/1GG5dYLfjTI_7907ORQJUX6q5Mg4wLI3U/view?usp=drive_link


## Training
You can train the models with the following codes:

For ERC task: 

`cd DAG-ECPE_ERC
python run.py --args*` 

The details of the parameters are: 

IEMOCAP: gnn_layers：1 ,batch_size：8 ,dropout：0.1,lr：0.0004,epoch：50 

MELD: gnn_layers：1 ,batch_size：8 ,dropout：0.2,lr：4e-5,epoch：30

DailyDialog:gnn_layers：1 ,batch_size：16 ,dropout：0.3,lr：5e-5,epoch：40

EmoryNLP:gnn_layers：1 ,batch_size：64 ,dropout：0.1,lr：7e-4,epoch：50

For ECPE 

`cd DAG-ECPE 
python run.py --args*` 

The details of the parameters are: 
lr:6e-4 

For ECSR 

`cd DAG-ECSR
python run.py --args*` 

The details of parameters are: 
lr:6e-4 
