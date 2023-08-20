# -*- coding: utf-8 -*-
"""
@Time:Created on 2019/5/20 20:49
@author: LiFan Chen
@Filename: main_glu.py
@Software: PyCharm
"""
import pickle

import torch
import numpy as np
import random
import os
import time
from model_glu import *
import timeit
import argparse

import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def load_tensor(file_name, dtype):
    return [dtype(d).to(device) for d in np.load(file_name + '.npy',allow_pickle=True)]


def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset


def split_dataset(dataset, ratio):
    n = int(ratio * len(dataset))
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2


def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)

if __name__ == "__main__":
    SEED = 1  # 1234
    random.seed(SEED)
    torch.manual_seed(SEED)
    # torch.backends.cudnn.deterministic = True
    # DATASET = "C.elegans"
    DATASET = "human"
    """CPU or GPU"""
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print('The code uses GPU...')
    else:
        device = torch.device('cpu')
        print('The code uses CPU!!!')

    """Load preprocessed data."""
    dir_input = ('dataset/' + DATASET + '/word2vec_30-modify-MDL-CPI/')
    compounds1 = load_tensor(dir_input + 'compounds1', torch.FloatTensor)
    compounds2 = load_tensor(dir_input + 'compounds2', torch.LongTensor)
    adjacencies = load_tensor(dir_input + 'adjacencies', torch.FloatTensor)
    proteins1 = load_tensor(dir_input + 'proteins1', torch.FloatTensor)
    proteins2 = load_tensor(dir_input + 'proteins2', torch.LongTensor)
    interactions = load_tensor(dir_input + 'interactions', torch.LongTensor)

    fingerprint_dict = load_pickle(dir_input + 'fingerprint_dict.pickle')
    word_dict = load_pickle(dir_input + 'word_dict.pickle')

    global n_fingerprint,n_word

    n_fingerprint = len(fingerprint_dict)
    n_word = len(word_dict)

    """Create a dataset and split it into train/dev/test."""
    dataset = list(zip(compounds1,compounds2, adjacencies, proteins1,proteins2, interactions))
    dataset = shuffle_dataset(dataset, 1234)    #  1234
    dataset_train, dataset_2= split_dataset(dataset, 0.8)
    dataset_dev, dataset_test = split_dataset(dataset_2, 0.5)

    """ create model ,trainer and tester """
    protein_dim = 100
    atom_dim = 34
    hid_dim = 64
    n_layers = 3
    n_heads = 8

    gat_heads = 3
    alpha = 0.2
    radius = 2
    ngram = 3

    pf_dim = 256
    dropout = 0.1  # 0.1
    batch = 32 # 64
    lr = 1e-3 # 1e-3
    weight_decay = 1e-4 # 1e-4
    decay_interval = 5 # 5
    # decay_interval1 = 10
    # decay_interval2 = 5
    # lr_decay = 0.8 # 0.5
    # lr_decay1 = 0.5
    # lr_decay2 = 0.8
    lr_decay = 0.5
    iteration = 40
    kernel_size = 5

    k_feature = 16  # drug和protein隐含关系的种类数，16
    k_dim = 16   # tensor_neurons,16

    epoch = 40


    gat = GAT(atom_dim, hid_dim,gat_heads, dropout, alpha, n_layers,device)
    bert = BERT(len(word_dict), device)
    inter_att = InteractionModel(hid_dim, n_heads)
    tensor_network = TensorNetworkModule(k_feature,hid_dim,k_dim)   # NTN
    # encoder = Encoder(protein_dim, hid_dim, 3, kernel_size, dropout, device)
    decoder = Decoder(atom_dim, hid_dim, n_layers, n_heads, pf_dim, DecoderLayer, SelfAttention, PositionwiseFeedforward, dropout, device)    # IMT
    model = Predictor(gat,bert, decoder, inter_att,tensor_network,device,n_fingerprint,n_layers)

    # model.load_state_dict(torch.load("output/model/lr=0.001,dropout=0.1,lr_decay=0.5"))
    model.to(device)
    trainer = Trainer(model, lr, weight_decay, batch)
    tester = Tester(model)

    """Output files."""
    file_AUCs = 'output_human/result-modify-7/AUCs--lr=1e-3,dropout=0.1,weight_decay=1e-4,k_feature=16,n_layer=3,batch=32,eopch=40,decay_interval=5.txt'
    file_model = 'output_human/model-modify-7/lr=1e-3,dropout=0.1,weight_decay=1e-4,k_feature=16,n_layer=3,batch=32,epoch=40,decay_interval=5.pt'
    # file_AUCs = 'output_C.elegans/result-C.elegans/AUCs--lr=1e-3,dropout=0.1,weight_decay=1e-4,k_feature=16,n_layer=3,batch=32,decay_interval=5.txt'
    # file_model = 'output_C.elegans/model-C.elegans/lr=1e-3,dropout=0.1,weight_decay=1e-4,k_feature=16,n_layer=3,batch=32,decay_interval=5.pt'
    AUCs = ('Epoch\tTime(sec)\tLoss_train\tAUC_dev\tAUC_test\tPrecision_test\tRecall_test')
    with open(file_AUCs, 'w') as f:
        f.write(AUCs + '\n')

    """Start training."""
    print('Training...')
    print(AUCs)
    start = timeit.default_timer()

    #max_AUC_dev = 0
    max_AUC_test = 0
    epoch_label = 0
    for epoch in range(1, epoch+1):
        # if epoch % decay_interval == 0:
        #     trainer.optimizer.param_groups[0]['lr'] *= lr_decay
        # if epoch >= 15 and epoch <=30 and epoch % decay_interval == 0:
        #     trainer.optimizer.param_groups[0]['lr'] *= lr_decay

        # 动态调整LR：前20个epoch，每10个epoch调整0.5，后20个epoch，每5个epoch调整0.5
        # if epoch < 20:
        #     if epoch % decay_interval1 == 0:                           # decay_interval1 = 10
        #         trainer.optimizer.param_groups[0]['lr'] *= lr_decay1   # lr_decay1 = 0.5
        # else:
        #     if epoch % decay_interval2 == 0:                           # decay_interval2 = 5
        #         trainer.optimizer.param_groups[0]['lr'] *= lr_decay1   # lr_decay2 = 0.5


        loss_train = trainer.train(dataset_train, device)
        AUC_dev,_,_ = tester.test(dataset_dev)
        AUC_test, precision_test, recall_test = tester.test(dataset_test)

        end = timeit.default_timer()
        time = end - start

        AUCs = [epoch, time, loss_train, AUC_dev, AUC_test, precision_test, recall_test]
        tester.save_AUCs(AUCs, file_AUCs)
        if  AUC_test > max_AUC_test:
            tester.save_model(model, file_model)
            max_AUC_test = AUC_test
            epoch_label = epoch
        print('\t'.join(map(str, AUCs)))

    print("The best model is epoch",epoch_label)
