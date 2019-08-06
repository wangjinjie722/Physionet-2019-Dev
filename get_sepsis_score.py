#!/usr/bin/env python

import os
import sys
import copy
import random
import pickle
import zipfile
import numpy as np
import pandas as pd
from tqdm import tqdm
from os import listdir
import tensorflow as tf
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
from keras import backend as K
import matplotlib.pyplot as plt

# parameters

feature_to_use = ['custom_age', 'custom_hr', 'custom_o2sat', 'custom_temp', 
                 'custom_resp', 'custom_hco3', 'custom_ph',
                  'custom_bun', 'custom_BC_ratio', 'qSOFA', 'SOFA' ,'SOFA_score',
                 'custom_calcium', 'custom_chloride', 'custom_creatinine',
                 'custom_glucose', 'custom_magnesium', 'custom_phosphate', 'custom_potassium',
                 'custom_hct', 'custom_hgb', 'custom_wbc', 
                  'custom_platelets', 'Gender', 'Unit', 'HospAdmTime','SepsisLabel']

custom = ['custom_age', 'custom_hr', 'custom_o2sat', 'custom_temp', 
                 'custom_resp', 'custom_hco3', 'custom_ph',
                  'custom_bun', 'custom_BC_ratio', 'qSOFA', 'SOFA' ,'SOFA_score',
                 'custom_calcium', 'custom_chloride', 'custom_creatinine',
                 'custom_glucose', 'custom_magnesium', 'custom_phosphate', 'custom_potassium',
                 'custom_hct', 'custom_hgb', 'custom_wbc', 
                  'custom_platelets', 'Gender', 'Unit', 'HospAdmTime']

# Missing:0 normal:1 abnormal:2 low:3 high:6 serious:5 moderate:4 elevated:7
# infant:1 adult:2 old:3

def feature_engineer_cut(train):
    '''
        unit every patient to 100 time length
        input: dataframe for one patient
        output: dataframe for the same patient with 100 time length

                use .copy to avoid copying warning
    '''
    tmp = train.tail(1)
    if len(train) < 100:
        train = train.append([tmp] * (100 - len(train) - 1))
        return train.copy()
    elif len(train) >= 100:
        return train[:99].copy()

def feature_engineer(train):
    '''
        create new features

        custom = ['custom_age', 'custom_hr', 'custom_o2sat', 'custom_temp', 
                 'custom_resp', 'custom_hco3', 'custom_ph',
                  'custom_bun', 'custom_BC_ratio', 'qSOFA', 'SOFA' ,'SOFA_score',
                 'custom_calcium', 'custom_chloride', 'custom_creatinine',
                 'custom_glucose', 'custom_magnesium', 'custom_phosphate', 'custom_potassium',
                 'custom_hct', 'custom_hgb', 'custom_wbc', 
                  'custom_platelets', 'Gender', 'Unit', 'HospAdmTime']
                
        up-to-date feature we will use
        08/06 - kw - add some sofa features
    '''
    train.fillna(method='pad', inplace = True)

    # zhaoweizhu 08/05/2019 new method
    # for index,col in train.iterrows():
    #     train.at[index,'col_name']=new_value#更改值
    # col['new_col']=new_col_value


    # hr
    train.loc[((train['HR'] >= 100) & (train['Age'] >= 10 ))
         | ((train['HR'] < 70) | (train['HR'] >= 190)) & (train['Age'] < 10 ),'custom_hr'] = 2
    train.loc[((train['HR'] < 100) & (train['HR'] > 60) & (train['Age'] >= 10 )) 
         | ((train['HR'] >= 70) & (train['HR'] < 190) & (train['Age'] < 10 )),'custom_hr'] = 1
    
    # temp
    train.loc[(train['Temp'] >= 36.4) & (train['Temp'] < 37.6), 'custom_temp'] = 1
    train.loc[(train['Temp'] < 36.4) | (train['Temp'] >= 37.6), 'custom_temp'] = 2

    # age
    train.loc[train['Age'] >=65, 'custom_age'] = 3
    train.loc[train['Age'] <10, 'custom_age'] = 1
    train.loc[(train['Age'] >=10) & (train['Age'] <65), 'custom_age'] = 2

    # o2sat
    train.loc[(train['O2Sat'] >= 90) & (train['O2Sat'] < 100), 'custom_o2sat'] = 1
    train.loc[(train['O2Sat'] < 90) & (train['O2Sat'] >= 0), 'custom_o2sat'] = 2

    # resp_rate
    train.loc[((train['Resp'].between(30,60)) & (train['Age'] <1)) 
        | ((train['Resp'].between(24,40)) & (train['Age'].between(1,3)))
        | ((train['Resp'].between(22,34)) & (train['Age'].between(3,6)))
        | ((train['Resp'].between(18,30)) & (train['Age'].between(6,12)))
        | ((train['Resp'].between(12,20)) & (train['Age'] >12)), 'custom_resp'] = 1

    train.loc[(((train['Resp'] < 30) | (train['Resp'] > 60)) & (train['Age'] <1)
        | (((train['Resp'] < 24) | (train['Resp'] > 40)) & (train['Age'].between(1,3))))
        | (((train['Resp'] < 22) | (train['Resp'] > 34)) & (train['Age'].between(3,6)))
        | (((train['Resp'] < 18) | (train['Resp'] > 30)) & (train['Age'].between(6,12)))
        | (((train['Resp'] < 12) | (train['Resp'] > 20)) & (train['Age'] >12)) ,'custom_resp'] = 2

    # ph
    feature = 'pH'
    train.loc[(train[feature] >= 7.35) & (train[feature] <= 7.45), 'custom_ph'] = 1
    train.loc[(train[feature] < 7.35) | (train[feature] > 7.45), 'custom_ph'] = 2

    # hco3
    feature = 'HCO3'
    train.loc[(train[feature] >= 22) & (train[feature] <= 27), 'custom_hco3'] = 1
    train.loc[(train[feature] < 22) | (train[feature] > 27), 'custom_hco3'] = 2

    # bun
    feature = 'BUN'
    up = 21
    down = 7
    custom = 'custom_' + feature.lower()
    train.loc[(train[feature] >= down) & (train[feature] <= up), 'custom_' + feature.lower()] = 1
    train.loc[(train[feature].between(up, 60)), 'custom_' + feature.lower()] = 2
    train.loc[(train[feature] < down), 'custom_' + feature.lower()] = 3
    train.loc[(train[feature] >= 60), 'custom_' + feature.lower()] = 4
    train.loc[(train['BUN'] != None) & (train['Creatinine'] != None) & ((train['BUN']/train['Creatinine'] >= 10) 
        | (train['BUN']/train['Creatinine'] <= 20)), 'custom_BC_ratio'] = 1
    
    # creatinine
    feature = 'Creatinine'
    up = 0.7
    down = 1.2
    up_ = 0.5
    down_ = 1.0
    custom = 'custom_' + feature.lower()
    train.loc[((train[feature] >= down) & (train[feature] <= up) & (train['Gender'] == 1))
        | ((train[feature] >= down_) & (train[feature] <= up_) & (train['Gender'] == 0)), 'custom_' + feature.lower()] = 1
    train.loc[(((train[feature] < down) | (train[feature] > up)) & (train['Gender'] == 1))
        | (((train[feature] < down_) | (train[feature] > up_)) & (train['Gender'] == 0)), 'custom_' + feature.lower()] = 2

    # calcium
    feature = 'Calcium'
    up = 10.2
    down = 8.5
    custom = 'custom_' + feature.lower()
    train.loc[(train[feature] >= down) & (train[feature] <= up), 'custom_' + feature.lower()] = 1
    train.loc[(train[feature] < down) | (train[feature] > up), 'custom_' + feature.lower()] = 2

    # chloride
    feature = 'Chloride'
    up = 98
    down = 106
    custom = 'custom_' + feature.lower()
    train.loc[(train[feature] >= down) & (train[feature] <= up), 'custom_' + feature.lower()] = 1
    train.loc[((train[feature] >= 70) & (train[feature] < down)) | (train[feature] > up & (train[feature] <= 120)), 'custom_' + feature.lower()] = 2
    train.loc[(train[feature] < 70) | (train[feature] > 120), 'custom_' + feature.lower()] = 4 

    # glucose
    feature = 'Glucose'
    up = 125
    down = 100
    custom = 'custom_' + feature.lower()
    train.loc[(train[feature] >= down) & (train[feature] <= up), 'custom_' + feature.lower()] = 1
    train.loc[(train[feature] < down) | (train[feature].between(up, 200)), 'custom_' + feature.lower()] = 2
    train.loc[(train[feature] >= 200), 'custom_' + feature.lower()] = 4
    
    # magnesium
    feature = 'Magnesium'
    up = 1.1
    down = 0.6
    custom = 'custom_' + feature.lower()
    train.loc[(train[feature] >= down) & (train[feature] <= up), 'custom_' + feature.lower()] = 1
    train.loc[(train[feature].between(up,2.9)), 'custom_' + feature.lower()] = 2
    train.loc[(train[feature] < down), 'custom_' + feature.lower()] = 3
    train.loc[(train[feature] >= 2.9) , 'custom_' + feature.lower()] = 4

    # phosphate
    feature = 'Phosphate'
    up = 4.5
    down = 2.5
    custom = 'custom_' + feature.lower()
    train.loc[(train[feature] >= down) & (train[feature] <= up), 'custom_' + feature.lower()] = 1
    train.loc[(train[feature] < down) | (train[feature] > up), 'custom_' + feature.lower()] = 2

    # potassium
    feature = 'Potassium'
    up = 5.2
    down = 3.6
    custom = 'custom_' + feature.lower()
    train.loc[(train[feature] >= down) & (train[feature] <= up), 'custom_' + feature.lower()] = 1
    train.loc[(train[feature] < down) | (train[feature] > up) & (train[feature] <= 6), 'custom_' + feature.lower()] = 2
    train.loc[(train[feature] > 6), 'custom_' + feature.lower()] = 4

    # hct
    feature = 'Hct'
    up = 52
    down = 45
    up_ = 48
    down_ = 37
    custom = 'custom_' + feature.lower()
    train.loc[((train[feature] >= down) & (train[feature] <= up) & (train['Gender'] == 1))
        | ((train[feature] >= down_) & (train[feature] <= up_) & (train['Gender'] == 0)), 'custom_' + feature.lower()] = 1
    train.loc[(((train[feature] < down) | (train[feature] > up)) & (train['Gender'] == 1))
        | (((train[feature] < down_) | (train[feature] > up_)) & (train['Gender'] == 0)), 'custom_' + feature.lower()] = 2

    # hgb
    feature = 'Hgb'
    up = 17.5
    down = 13.5
    up_ = 15.5
    down_ = 12
    custom = 'custom_' + feature.lower()
    train.loc[((train[feature] >= down) & (train[feature] <= up) & (train['Gender'] == 1))
        | ((train[feature] >= down_) & (train[feature] <= up_) & (train['Gender'] == 0)), 'custom_' + feature.lower()] = 1
    train.loc[(((train[feature] < down) | (train[feature] > up)) & (train['Gender'] == 1))
        | (((train[feature] < down_) | (train[feature] > up_)) & (train['Gender'] == 0)), 'custom_' + feature.lower()] = 2

    # wbc
    feature = 'WBC'
    up = 11
    down = 4.5
    custom = 'custom_' + feature.lower()
    train.loc[(train[feature] >= down) & (train[feature] <= up), 'custom_' + feature.lower()] = 1
    train.loc[(train[feature] < down) | (train[feature] > up), 'custom_' + feature.lower()] = 2

    # platelets
    feature = 'Platelets'
    up = 400
    down = 150
    custom = 'custom_' + feature.lower()
    train.loc[(train[feature] >= down) & (train[feature] <= up), 'custom_' + feature.lower()] = 1
    train.loc[(train[feature] < down) | (train[feature] > up), 'custom_' + feature.lower()] = 2

    # unit
    train.loc[train['Unit1'] == 1, 'Unit'] = 1
    train.loc[train['Unit2'] == 1, 'Unit'] = 2

    # Gender
    train.loc[train['Gender'] == 0, 'Gender'] = 2
    
    # SOFA
    train.loc['SOFA'] = 0
    train.loc[train['Platelets'].between(150, 99),'SOFA'] = 1
    train.loc[train['Platelets'].between(100, 50),'SOFA'] = 2
    train.loc[train['Platelets'].between(49, 20),'SOFA'] = 3
    train.loc[train['Platelets'].between(19, 1),'SOFA'] = 4
    train['SOFA'].fillna(0, inplace = True)

    train.loc[train['Bilirubin_direct'].between(1.2, 1.9),'SOFA'] += 1
    train.loc[train['Bilirubin_direct'].between(2, 5.9),'SOFA'] += 2
    train.loc[train['Bilirubin_direct'].between(6, 11.9),'SOFA'] += 3
    train.loc[train['Bilirubin_direct'] > 12,'SOFA'] += 4
    
    train.loc[train['Creatinine'].between(1.2, 1.9),'SOFA'] += 1
    train.loc[train['Creatinine'].between(2, 3.4),'SOFA'] += 2
    train.loc[train['Creatinine'].between(3.5, 4.9),'SOFA'] += 3
    train.loc[train['Creatinine'] > 5,'SOFA'] += 4
    
    # SOFA_socre
    train.loc[train['SOFA'] > 2,'SOFA_score'] = 2 # abnormal
    train.loc[train['SOFA'] <= 2,'SOFA_score'] = 1 # normal
   
    # qSOFA
    
    train.loc[(train['Resp'] > 22) & (train['SBP'] < 100),'qSOFA'] = 2 # abnormal
    train.loc[(train['Resp'] <= 22) | (train['SBP'] >= 100),'qSOFA'] = 1 # normal
    
    train.fillna(0, inplace = True)
    
    return train[feature_to_use].copy()

def feature_engineering(train):
    '''
        feature engineering function
        input: dataframe meta_data for one patient
        output: dataframe for the same patient with new features and 100 time length
    '''
    train = feature_engineer_cut(train)
    train = feature_engineer(train)
    return train

def w_binary_crossentropy(y_true, y_pred):
        return K.mean(tf.nn.weighted_cross_entropy_with_logits(
                                                               y_true,
                                                               y_pred,
                                                               weights,
                                                               name=None
                                                               ), axis=-1)

def weighted_binary_crossentropy(weights):
    '''
        self-defined loss function (weighted binary crossentropy)
        input: weights
        output: loss function with weight
    '''
    def w_binary_crossentropy(y_true, y_pred):
        return K.mean(tf.nn.weighted_cross_entropy_with_logits(
                                                               y_true,
                                                               y_pred,
                                                               weights,
                                                               name=None
                                                               ), axis=-1)
    return w_binary_crossentropy

# wbc loss function used to make the model relys more on sepsis samples
kw_loss = weighted_binary_crossentropy(weights=1)

# the four functions blow are used for metrics, f-measure is also used in offical check
def precision(y_true, y_pred):
    # Calculates the precision
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    # Calculates the recall
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def fbeta_score(y_true, y_pred, beta=1):
    # Calculates the F score, the weighted harmonic mean of precision and recall.
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')
    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score

def fmeasure(y_true, y_pred):
    # Calculates the f-measure, the harmonic mean of precision and recall.
    return fbeta_score(y_true, y_pred, beta=1)

def fix_100(res, org_length):
    """
        This is a function that convert 100 time long sequence to its original length that can be checked by utility score.
        input: 
                res is the output of your model
                org_length is the original length of each patient
                threshold is defined as global
        output:
                sepsis probabilities and predictions, of course in original length
    """
    l = 100 # fixed length of each patient
    # inilization for probabilities and predictions with original length
    tmp_res = list(res)
    if org_length <= l:
        tmp_res = tmp_res[:org_length]
    else:
        last = tmp_res[-1]
        tmp_res += [last for _ in range(org_length - l)]

    tmp_predict = [1 for _ in range(len(tmp_res))]

    for r in range(len(tmp_res)):
        if tmp_res[r] < threshold:
            tmp_predict[r] = 0
        else:
            tmp_res_ = tmp_res[:r] + [tmp_res[r] for _ in range(len(tmp_res) - r)]  
            return tmp_res_, tmp_predict

    return tmp_res, tmp_predict

def load_sepsis_model():
    
    #model = load_model(model_name) # you need this without self defined loss function
    model = load_model(model_name, custom_objects={'w_binary_crossentropy':w_binary_crossentropy,'fmeasure':fmeasure}) # only for self-defined weighted binary crossentropy situation
    return model

# change me everytime you change the model
NOW = '2019-08-06-10-40-04'
model_name = f'./model/LSTM_{NOW}.h5'
threshold = 0.5

def get_sepsis_score(data, model):
    
    # load your model
    LSTM_model = model
    # read meta_data in psv format
    meta_data = pd.read_csv(data, sep='|')
    # get orginal length that can be used to convert to its orginal
    org_length = len(meta_data)
    #print(org_length)
    # feature engineering
    cur_train = feature_engineering(meta_data)
    #print(len(cur_train))
    # define weighted loss that maybe use, you can ignore it if you are not gonna use it
    kw_loss = weighted_binary_crossentropy(weights=1)

    # inilization for the save of result
    preds = [0 for _ in range(100)]
    org_pred = [0 for _ in range(100)]
    tmp = [0 for _ in range(100)]

    # process the test data
    dtest = np.array(cur_train[custom])
    dtest = dtest.reshape(-1,100,len(custom))

    # change me if you use different kind of model
    predicted = list(LSTM_model.predict(dtest)[0]) # if you use reshape to make it 3-dim, you need [0] here trust me :-)
    
    # save the true label
    org_label = cur_train['SepsisLabel']

    for t in range(100):
        # this makes predictions really slow and this is the way for offical check stage
        # blow code is used to get the official drive.py work

    #    cur_train = feature_engineering(cur_train)[feature_to_use]
    #    dtest = np.array(cur_train[custom][:t+1])
    #    dtest = dtest.reshape(-1,100,len(custom))
    #    predicted = list(LSTM_model.predict(dtest)[0])

        preds[t] = predicted[t]

        # This is the correction method that can deal with the high false positive rate
        
        front_preds = []
        if t >= 1:
            front_preds.append(predicted[t-1])
        if t >= 2:
            front_preds.append(predicted[t-2])
        if t >= 3:
            front_preds.append(predicted[t-3])

        tmp_fp = []
        tmp_p = []

        for fp in front_preds:
            if fp >= threshold:
                tmp_fp.append(1)
                tmp_p.append(fp)
            elif fp < threshold:
                tmp_fp.append(0)
                tmp_p.append(1-fp)
        tmp[t] = 0
        for i in range(len(tmp_p)):
            tmp[t] += (tmp_p[i] * tmp_fp[i])
        org_pred[t] = preds[t]
        if len(tmp_p) >= 2:
            tmp[t] /= len(tmp_p)
            # change this two weight to change the correction method
            preds[t] = org_pred[t] * 0.8 + 0.2 * tmp[t]
        else:
            preds[t] = org_pred[t]

    preds = [p if p > 0 else 0 for p in preds]
    preds_max = max(preds)
    preds = [x / (preds_max + 0.2) for x in preds]
    #label = [1 if p >= threshold else 0 for p in preds]

    # these parameters are used to plot
    score, label = fix_100(preds, org_length) # probabilities and predicted label required by offical check
    org_pred,_ = fix_100(org_pred, org_length) # this is the original predicted probabilities given by model
    tmp,_ = fix_100(tmp, org_length) # this is the correction value given by the front 3 points
    org_label, _ = fix_100(org_label, org_length) # this is the true label

    # you can change the output if you like
    return score, label, org_pred, tmp

