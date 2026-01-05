## AMMVF
An end-to-end deep learning method (AMMVF) is proposed to predict DTIs based on drug-target independent features and interaction features from both node-level and graph-level embeddings.

## Requirements
* python >= 3.8
* torch >= 1.11
* CUDA >= 11.3
* RDkit >= 2020.09.1
* numpy >= 1.21.5
* pandas >= 1.3.5

## Use
1. main_glu.py: a start file for model training
2. model_glu.py: the construction of neural network
3. mol_featurizer.py: data processing to get the input of the model

## Authors
This code was originally created by Lu Wang, who is currently a master student in Zhejiang University of Science and Technology.
This code is derived from her website: https://github.com/xiaoluobie/AMMVI.

Lu Wang is under joint supervision of Dr. Qu Chen and Prof. Yifeng Zhou.
This code serves as the Supporting Information for the manuscript entitled "AMMVF-DTI: A Novel Model Predicting Drug-Target Interactions Based on Attention Mechanism and Multi-View Fusion, Int. J. Mol. Sci., 2023, 24, 14142" and can be downloaded for free.

edited on September 9th, 2023
