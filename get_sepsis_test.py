#!/usr/bin/env python
import os
import sys
import copy
import time
import random
import pickle
import zipfile
import numpy as np
import pandas as pd
from tqdm import tqdm
from os import listdir
from keras import layers
from keras import models
from sklearn import metrics
from keras.layers import LSTM
from keras.layers import Dense
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.models import Sequential
from sklearn.metrics import roc_auc_score
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder,MinMaxScaler
from sklearn.metrics import confusion_matrix, precision_score, recall_score, \
    average_precision_score, precision_recall_curve

from get_sepsis_score import *


#读取文件
def read_txt(txt_name):
    patient = []
    rs = os.path.exists(txt_name)
    if rs:
        with open(txt_name,mode='r') as file_handler:
            contents = file_handler.readlines()
            for name in contents:
                name = name.strip('\n')
                patient.append(name)
    else:
        print('txt does not exit')
    return patient


if __name__ == '__main__':

    # inilization
    MODEL = load_sepsis_model()
    patient_a = read_txt('/Users/wangkai/Downloads/PN2019/patient_a.txt')
    patient_b = read_txt('/Users/wangkai/Downloads/PN2019/patient_b.txt')
    score = []
    label = []
    org_label = []
    org_length = []

    # pkl_file = open(sys.argv[1], 'rb')
    # test_candidate = pickle.load(pkl_file)
    # pkl_file.close()

    test_dir = listdir('/Users/wangkai/Downloads/PN2019/data/trainingB/')
    random.shuffle(test_dir)
    random.shuffle(patient_a)
    random.shuffle(patient_b)
    test_candidate = test_dir[:int(sys.argv[1])]
    test_candidate += patient_b[:int(int(sys.argv[1]) // 20)]
    random.shuffle(test_candidate)
    org = []
    # test every case
    for i in tqdm(test_candidate):
        i = '/Users/wangkai/Downloads/PN2019/data/trainingB/' + i
        tmp_data = pd.read_csv(i, sep='|')
        single_score, single_label, single_org = get_sepsis_score(i, MODEL)

        if not len(org_label):
            org_label = list(tmp_data['SepsisLabel'])
        else:
            org_label += list(tmp_data['SepsisLabel'])

        if not len(score):
            score = list(single_score)
        else:
            score += list(single_score)

        if not len(label):
            label = list(single_label)
        else:
            #print(len(label), len(single_label))
            #print(type(label), type(single_label))
            label += list(single_label)


        if not len(org):
            org = list(single_org)
        else:
            org += list(single_org)

    # save results
    with open('labels.txt', 'w') as f:
        f.write('SepsisLabel\n')
        if len(org_label) != 0:
            for l in org_label:
                f.write('%d\n' % l)
    with open('predictions.txt', 'w') as f:
        f.write('PredictedProbability|PredictedLabel\n')
        if len(score) != 0:
            for (s, l) in zip(score, label):
                f.write('%g|%d\n' % (s, l))

    # make zipfiles 
    with zipfile.ZipFile('labels.zip', 'w') as z:
        z.write('labels.txt')
    with zipfile.ZipFile('predictions.zip', 'w') as z:
        z.write('predictions.txt')   
    
    plt.figure(1)
    plt.scatter(list(range(len(org_label))), score,c = 'g',alpha = 0.1)
    plt.scatter(list(range(len(org_label))), org_label,c = 'b',alpha = 0.1)
    plt.scatter(list(range(len(org_label))), label,c = 'r',alpha = 0.1)
    plt.legend(['probability','true','prdicted',], loc = 'lower right')
    plt.title('Predicted Result of N samples from set B by model trained by set A')
    plt.show()

#    plt.figure(2)
#    plt.scatter(list(range(len(org))), org,c = 'y',alpha = 0.1)
#    plt.show()

    # python3 evaluate_sepsis_score.py labels.zip predictions.zip


