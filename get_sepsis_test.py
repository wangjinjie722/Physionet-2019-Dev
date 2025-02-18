#!/usr/bin/env python

import os
import sys
import copy
import time
import random
import pickle
import zipfile
import shutil
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
import tensorflow as tf
from get_sepsis_score import *

# you can fix the test case order if you like
#random.seed(7)

if __name__ == '__main__':

    # inilization
    MODEL = load_sepsis_model()

    score = [] # predictied probability
    label = [] # predicted label
    org_label = [] # original label
    org_length = [] # original time length for patient
    org_pred = [] # current point's prediction
    top3 = [] # 3 points' predictions in front
    
    # load patient list if you like
    try:
        with open('./data/patient.pkl', 'rb') as file:
            patient = pickle.load(file)
        with open('./data/normal.pkl', 'rb') as file:
            normal = pickle.load(file)

        with open('./data/test2.pkl', 'rb') as file:
            test2 = pickle.load(file)
    except:
        print('you need get the pkls from dachui wang or github')

    
    # shuffle it and get it randomly
    random.shuffle(patient)
    random.shuffle(normal)

    # this is the options that you can find the details in readme.md
    # 0 代表全是正常人 
    # - 1 代表全是病人 
    # - 2 代表比例为7.8%的病人正常人混合样本 
    # - 3 需要在数量位置传入新参数trainingA/p019094.psv 代表预测当前特指的样本
    # - 11 测试集1含有1000个随机出的样本
    # - 22 测试集2含有1000个随机出的样本
    # - 55 测试集3含有500个随机出的样本
    # - 555 测试集4含有500个随机出的样本
    # - 911 test the same cases as official drive.py

    if int(sys.argv[1]) == 1:
        test_candidate = patient[:int(int(sys.argv[2]) // 1)]
        print(test_candidate)
    elif int(sys.argv[1]) == 0:
        test_candidate = normal[:int(sys.argv[2])]
        print(test_candidate)
    elif int(sys.argv[1]) == 2:
        test_candidate = normal[:int(sys.argv[2])]
        test_candidate += patient[:round(int(sys.argv[2]) / 12.75)]
        print(test_candidate)
    elif int(sys.argv[1]) == 3:
        test_candidate = [sys.argv[2]]
        print(test_candidate)
    elif int(sys.argv[1]) == 11:
        with open('./data/test1.pkl', 'rb') as file:
            test_candidate = pickle.load(file)
    elif int(sys.argv[1]) == 22:
        with open('./data/test2.pkl', 'rb') as file:
            test_candidate = pickle.load(file)
    elif int(sys.argv[1]) == 55:
        with open('./data/test3.pkl', 'rb') as file:
            test_candidate = pickle.load(file)
    elif int(sys.argv[1]) == 555:
        with open('./data/test4.pkl', 'rb') as file:
            test_candidate = pickle.load(file)
    elif int(sys.argv[1]) == 911:
        tmp_candidate = listdir('./data/test')
        if '.DS_Store' in tmp_candidate:
            tmp_candidate.remove('.DS_Store')
        test_candidate = ['trainingA/' + l for l in tmp_candidate]
    # shuffle it again
    random.shuffle(test_candidate)

    # empty the result dir everytime you predict
    predict_dir = './predictions'
    label_dir = './labels'
    shutil.rmtree(predict_dir)
    os.mkdir(predict_dir)
    shutil.rmtree(label_dir)
    os.mkdir(label_dir)
    
    # test every case
    for i in tqdm(test_candidate):
    
        name = i[10:17]
        i = './data/' + i
        tmp_data = pd.read_csv(i, sep='|')
        single_org = tmp_data['SepsisLabel']

        # you need to change the predict sentence as your predict function
        # input should be the name of test case like './data/trainingA/p000100.psv'
        single_score, single_label, single_pred, single_top3 = get_sepsis_score(i, MODEL)
        
        # save results
        # name is the number of patient like 'p000100'
        with open(f'./labels/{name}.txt', 'w') as f:
            f.write('SepsisLabel\n')
            # single_org is the true label
            if len(single_org) != 0:
                for l in list(single_org):
                    f.write('%d\n' % l)
                f.close()
    
        with open(f'./predictions/{name}.txt', 'w') as f:
            f.write('PredictedProbability|PredictedLabel\n')
            # single score is probabilities and label is predicted label
            if len(single_score) != 0:
                for (s, l) in zip(list(single_score), list(single_label)):
                    f.write('%g|%d\n' % (s, l))
            f.close()
        
        # save results for every point for plot
        if not len(org_pred):
            org_pred = list(single_pred)
        else:
            org_pred += list(single_pred)
        if not len(top3):
            top3 = list(single_top3)
        else:
            top3 += list(single_top3)
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
            label += list(single_label)   

    # make zipfiles 
    with zipfile.ZipFile('labels.zip', 'w') as z:
        for i in test_candidate:
            name = i[10:17]
            z.write(f'./labels/{name}.txt')
    with zipfile.ZipFile('predictions.zip', 'w') as z:
        for i in test_candidate:
            name = i[10:17]
            z.write(f'./predictions/{name}.txt')

    # plot part
    # make it more clear for true value and predicted value
    org_label = [ol + 0.05 if ol == 1 else ol - 0.05 for ol in org_label]

    plt.figure(1)
    plt.scatter(list(range(len(org_label))), score,c = 'g',alpha = 0.1) # probabilities
    plt.scatter(list(range(len(org_label))), org_label,c = 'b',alpha = 0.1) # true label
    plt.scatter(list(range(len(org_label))), label,c = 'r',alpha = 0.1) # predicted label
    plt.scatter(list(range(len(org_label))), org_pred,c = 'black',alpha = 0.1) # original prababilities on current point
    plt.scatter(list(range(len(org_label))), top3,c = 'y',alpha = 0.1) # corrections values for 3 points in front of current point
    
    plt.legend(['probability','true','prdicted','org_pred','top3'], loc = 'lower right')
    plt.title(f'Predicted Result of {len(test_candidate)} samples')
    plt.show()

