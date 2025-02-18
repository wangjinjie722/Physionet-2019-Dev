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

from get_sepsis_score import *


# load_txt
#def read_txt(txt_name):
#    patient = []
#    rs = os.path.exists(txt_name)
#    if rs:
#        with open(txt_name,mode='r') as file_handler:
#            contents = file_handler.readlines()
#            for name in contents:
#                name = name.strip('\n')
#                patient.append(name)
#    else:
#        print('txt does not exit')
#    return patient


#    patient_a = read_txt('/Users/wangkai/Downloads/PN2019/patient_a.txt')
#    patient_b = read_txt('/Users/wangkai/Downloads/PN2019/patient_b.txt')
#    test_dir = listdir('/Users/wangkai/Downloads/PN2019/data/trainingB/')
#    random.shuffle(test_dir)
#    random.shuffle(patient_a)
#    random.shuffle(patient_b)

if __name__ == '__main__':

    # inilization
    MODEL = load_sepsis_model()

    score = [] # predictied probability
    label = [] # predicted label
    org_label = [] # original label
    org_length = [] # original time length for patient
    org_pred = [] # current point's prediction
    top3 = [] # 3 points' predictions in front
    
    with open('./data/patient.pkl', 'rb') as file:
        patient = pickle.load(file)
    with open('./data/normal.pkl', 'rb') as file:
        normal = pickle.load(file)
    random.shuffle(patient)
    random.shuffle(normal)

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

    random.shuffle(test_candidate)

    # empty the result dir
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
        single_score, single_label, single_pred, single_top3 = get_sepsis_score(i, MODEL)
        
        # save results
        with open(f'./labels/{name}.txt', 'w') as f:
            f.write('SepsisLabel\n')
            if len(single_org) != 0:
                for l in list(single_org):
                    #print(l, list(single_org))
                    f.write('%d\n' % l)
                f.close()
    
        with open(f'./predictions/{name}.txt', 'w') as f:
            f.write('PredictedProbability|PredictedLabel\n')
            if len(single_score) != 0:
                for (s, l) in zip(list(single_score), list(single_label)):
                    f.write('%g|%d\n' % (s, l))
            f.close()
        
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

    org_label = [ol + 0.05 if ol == 1 else ol - 0.05 for ol in org_label]
    plt.figure(1)
    plt.scatter(list(range(len(org_label))), score,c = 'g',alpha = 0.1)
    plt.scatter(list(range(len(org_label))), org_label,c = 'b',alpha = 0.1)
    plt.scatter(list(range(len(org_label))), label,c = 'r',alpha = 0.1)
    plt.scatter(list(range(len(org_label))), org_pred,c = 'black',alpha = 0.1)
    plt.scatter(list(range(len(org_label))), top3,c = 'y',alpha = 0.1)
    plt.legend(['probability','true','prdicted','org_pred','top3'], loc = 'lower right')
    plt.title('Predicted Result of N samples from set B by model trained by set A')
    plt.show()


    # python3 evaluate_sepsis_score.py labels.zip predictions.zip


