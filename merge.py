#!/usr/bin/env python
import os
import sys
import copy
import time
import random
import pickle
import zipfile
from tqdm import tqdm
import numpy as np
import pandas as pd
from pandas import *
from pylab import mpl
from os import listdir
from keras import layers
from keras import models
from sklearn import metrics
from keras.layers import LSTM
from keras.layers import Dense
from keras.models import load_model
from keras.models import Sequential
from sklearn.metrics import roc_auc_score
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder,MinMaxScaler
from sklearn.metrics import confusion_matrix, precision_score, recall_score, \
    average_precision_score, precision_recall_curve


# parameters

feature_to_use = ['custom_age', 'custom_bp', 'custom_hr', 'custom_o2sat', 'custom_temp',
                  'custom_resp', 'custom_etco2', 'custom_hco3', 'custom_sao2', 'custom_ph',
                  'custom_ast', 'custom_bun', 'custom_BC_ratio', 'custom_alkalinephos',
                  'custom_calcium', 'custom_chloride', 'custom_creatinine', 'custom_bd', 'custom_bt',
                  'custom_glucose', 'custom_lactate', 'custom_magnesium', 'custom_phosphate', 'custom_potassium',
                  'custom_troponini', 'custom_hct', 'custom_hgb', 'custom_wbc', 'custom_fibrinogen',
                  'custom_platelets', 'Gender', 'Unit', 'HospAdmTime', 'ICULOS','SepsisLabel']

custom = ['custom_age', 'custom_hr', 'custom_o2sat', 'custom_temp',
          'custom_resp', 'custom_hco3', 'custom_ph',
          'custom_bun', 'custom_BC_ratio',
          'custom_calcium', 'custom_chloride', 'custom_creatinine',
          'custom_glucose', 'custom_magnesium', 'custom_phosphate', 'custom_potassium',
          'custom_hct', 'custom_hgb', 'custom_wbc',
          'custom_platelets', 'Gender', 'Unit', 'HospAdmTime', 'ICULOS']

custom_ = copy.deepcopy(custom)
custom_.remove('HospAdmTime')
custom_.remove('ICULOS')
model_name = 'LSTM_2019-07-25-10-30-44.h5'
encoder_name = 'Encoder_2019-07-25-10-30-44.pkl'
threshold = 0.5



if __name__ == '__main__':

    kind = '100'
    # A
    a_pkl = listdir(f'./data/A_{kind}/')
    trainA = [[] for _ in range(42)]

    for i in tqdm(range(40)):
        # print(f'set****************************{i}******************************')
        with open(f'./data/A_{kind}/A_{kind}_{i * 500}.pkl', 'rb') as file:
            trainA[i] = pickle.load(file)
        for _, pkl in enumerate(a_pkl[i * 500 + 1:(i + 1) * 500 + 1]):
            if pkl == '.DS_Store':
                continue
            with open(f'./data/A_{kind}/' + pkl, 'rb') as file:
                tmp = pickle.load(file)
                trainA[i] = pd.merge(trainA[i], tmp,how='outer')

    with open(f'./data/A_{kind}/A_{kind}_{20001}.pkl', 'rb') as file:
        trainA[40] = pickle.load(file)

    for _, pkl in tqdm(enumerate(a_pkl[40 * 500 + 2:])):
        with open(f'./data/A_{kind}/' + pkl, 'rb') as file:
            tmp = pickle.load(file)
            trainA[40] = pd.merge(trainA[40], tmp,how='outer')

    trainA[41] = trainA[0]
    with open(f"./data/A_100/A100.pkl", 'wb') as file_A:
        for i in tqdm(range(41)):
            tmp = trainA[i]
            trainA[41] = pd.merge(trainA[41], tmp,how='outer')
        pickle.dump(trainA[41], file_A)

    # B

#    b_pkl = listdir(f'./data/B_{kind}/')
#    trainB = [[] for _ in range(41)]
#    for i in tqdm(range(40)):
#        print(f'set****************************{i}******************************')
#        with open(f'./data/B_{kind}/B_{kind}_{i * 500}.pkl', 'rb') as file:
#            trainB[i] = pickle.load(file)
#
#        for _, pkl in enumerate(b_pkl[i * 500 + 1:(i + 1) * 500 + 1]):
#            if pkl == '.DS_Store':
#                continue
#            with open(f'./data/B_{kind}/' + pkl, 'rb') as file:
#                tmp = pickle.load(file)
#                trainB[i] = pd.merge(trainB[i], tmp,how='outer')
#
#    trainB[40] = trainB[0]
#    with open(f"./data/B_{kind}/B{kind}.pkl", 'wb') as file_B:
#        for i in tqdm(range(40)):
#            tmp = trainB[i]
#            trainB[40] = pd.merge(trainB[40], tmp,how='outer')
#        pickle.dump(trainB[40], file_B)
#
#    # AB
#    with open(f"./data/AB100.pkl", 'wb') as file_AB:
#        with open(f"./data/A_100/A100.pkl", 'rb') as file_A:
#            cutA = pickle.load(file_A)
#        with open(f"./data/B_100/B100.pkl", 'rb') as file_B:
#            cutB = pickle.load(file_B)
#
#        cut = pd.merge(cutA, cutB,how='outer')
#
#    pickle.dump(cut, file_AB)
#
