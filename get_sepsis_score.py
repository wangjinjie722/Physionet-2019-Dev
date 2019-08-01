#!/usr/bin/env python
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.logging.ERROR)

import os
import sys
import copy
import random
import pickle
import zipfile
import numpy as np
import pandas as pd
from pandas import *
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



# Missing:0 normal:1 abnormal:2 low:3 high:6 serious:5 moderate:4 elevated:7
# infant:1 adult:2 old:3
def feature_engineer_cut(train):
    tmp = train.tail(1)
    while len(train) <= 100:
        train = pd.concat([train,tmp],ignore_index=True,)
    if len(train) > 100:
        train = train[:100]
    return train

def feature_engineer_hr(train):
    train.loc[(train['HR'] >= 100) & (train['Age'] >= 10 ),'custom_hr'] = 2
    train.loc[(train['HR'] < 100) & (train['HR'] > 60) & (train['Age'] >= 10 ),'custom_hr'] = 1
    train.loc[(train['HR'] >= 70) & (train['HR'] < 190) & (train['Age'] < 10 ),'custom_hr'] = 1
    train.loc[((train['HR'] < 70) | (train['HR'] >= 190)) & (train['Age'] < 10 ),'custom_hr'] = 2
    train['custom_hr'] = train['custom_hr'].fillna(0)
    return train

def feature_engineer_temp(train):
    train.loc[(train['Temp'] >= 36.4) & (train['Temp'] < 37.6), 'custom_temp'] = 1
    train.loc[(train['Temp'] < 36.4) | (train['Temp'] >= 37.6), 'custom_temp'] = 2
    train['custom_temp'] = train['custom_temp'].fillna(0)
    return train

def feature_engineer_age(train):
    train.loc[train['Age'] >=65, 'custom_age'] = 3
    train.loc[train['Age'] <10, 'custom_age'] = 1
    train.loc[(train['Age'] >=10) & (train['Age'] <65), 'custom_age'] = 2
    return train

def feature_engineer_o2sat(train):
    train.loc[(train['O2Sat'] >= 90) & (train['O2Sat'] < 100), 'custom_o2sat'] = 1
    train.loc[(train['O2Sat'] < 90) & (train['O2Sat'] >= 0), 'custom_o2sat'] = 2
    train['custom_o2sat'] = train['custom_o2sat'].fillna(0)
    return train

def feature_engineer_blood_pressure(train):
    train.loc[(train['SBP'] <90) & (train['DBP'] <60), 'custom_bp'] = 3
    train.loc[(train['SBP'].between(90,120, inclusive=True)) & (train['DBP'].between(60,80, inclusive=True)), 'custom_bp'] = 1
    train.loc[(train['SBP'].between(120,140, inclusive=True)) & (train['DBP'].between(80,90, inclusive=True)), 'custom_bp'] = 7
    train.loc[(train['SBP'] > 140 ) & (train['DBP'] > 90 ), 'custom_bp'] = 6
    train['custom_bp'] = train['custom_bp'].fillna(0)
    return train

def feature_engineer_resp_rate(train):
    train.loc[(train['Resp'].between(30,60)) & (train['Age'] <1), 'custom_resp'] = 1
    train.loc[((train['Resp'] < 30) | (train['Resp'] > 60)) & (train['Age'] <1) ,'custom_resp'] = 2
    train.loc[(train['Resp'].between(24,40)) & (train['Age'].between(1,3)), 'custom_resp'] = 1
    train.loc[((train['Resp'] < 24) | (train['Resp'] > 40)) & (train['Age'].between(1,3)) ,'custom_resp'] = 2
    train.loc[(train['Resp'].between(22,34)) & (train['Age'].between(3,6)), 'custom_resp'] = 1
    train.loc[((train['Resp'] < 22) | (train['Resp'] > 34)) & (train['Age'].between(3,6)) ,'custom_resp'] = 2
    train.loc[(train['Resp'].between(18,30)) & (train['Age'].between(6,12)), 'custom_resp'] = 1
    train.loc[((train['Resp'] < 18) | (train['Resp'] > 30)) & (train['Age'].between(6,12)) ,'custom_resp'] = 2
    train.loc[(train['Resp'].between(12,20)) & (train['Age'] >12), 'custom_resp'] = 1
    train.loc[((train['Resp'] < 12) | (train['Resp'] > 20)) & (train['Age'] >12),'custom_resp'] = 2
    train['custom_resp'] = train['custom_resp'].fillna(0)
    return train

def feature_engineer_ph(train):
    feature = 'pH'
    custom = 'custom_ph'
    train.loc[(train[feature] >= 7.35) & (train[feature] <= 7.45), 'custom_ph'] = 1
    train.loc[(train[feature] < 7.35) | (train[feature] > 7.45), 'custom_ph'] = 2
    train[custom] = train[custom].fillna(0)
    return train

def feature_engineer_etco2(train):
    train.loc[(train['EtCO2'] >= 35) & (train['EtCO2'] <=45), 'custom_etco2'] = 1
    train.loc[(train['EtCO2'] < 35) | (train['EtCO2'] > 45), 'custom_etco2'] = 2
    train['custom_etco2'] = train['custom_etco2'].fillna(0)
    return train

def feature_engineer_baseexcess(train):
    feature = 'BaseExcess'
    train.loc[(train['BaseExcess'] >= -3) & (train['BaseExcess'] <= 3), 'custom_be'] = 1
    train.loc[(train['BaseExcess'] < -3) | (train['BaseExcess'] > 3), 'custom_be'] = 2
    train['custom_be'] = train['custom_be'].fillna(0)
    return train

def feature_engineer_hco3(train):
    feature = 'HCO3'
    train.loc[(train[feature] >= 22) & (train[feature] <= 27), 'custom_hco3'] = 1
    train.loc[(train[feature] < 22) | (train[feature] > 27), 'custom_hco3'] = 2
    train['custom_hco3'] = train['custom_hco3'].fillna(0)
    return train

def feature_engineer_fso2(train):
    feature = 'FiO2'
    feature_ = 'SaO2'
    custom = 'custom_fio2'
    custom_ = 'custom_sao2'
    
    train.loc[(train[feature] >= 0.18) & (train[feature] <= 0.25), custom] = 1
    train.loc[(train[feature] < 0.18) | (train[feature] > 0.25), custom] = 2
    
    train.loc[(train[feature_] >= 94) & (train[feature_] <= 100), custom_] = 1
    train.loc[(train[feature_] < 94) | (train[feature_] > 100), custom_] = 2
    
    train[custom] = train[custom].fillna(0)
    train[custom_] = train[custom_].fillna(0)
    return train

def feature_engineer_paco2(train):
    feature = 'PaCO2'
    custom = 'custom_paco2'
    train.loc[(train[feature] >= 38) & (train[feature] <= 42), 'custom_paco2'] = 1
    train.loc[(train[feature].between(42, 59)), 'custom_paco2'] = 5
    train.loc[(train[feature] >= 60), 'custom_paco2'] = 4
    train.loc[(train[feature] < 38), 'custom_paco2'] = 3
    
    train[custom] = train[custom].fillna(0)
    return train

def feature_engineer_ast(train):
    feature = 'AST'
    up = 34
    down = 6
    up_ = 40
    down_ = 8
    custom = 'custom_' + feature.lower()
    train.loc[(train[feature] >= down) & (train[feature] <= up) & (train['Gender'] == 1), 'custom_' + feature.lower()] = 1
    train.loc[(train[feature] < down) | (train[feature] > up) & (train['Gender'] == 1), 'custom_' + feature.lower()] = 2
    train.loc[(train[feature] >= down_) & (train[feature] <= up_) & (train['Gender'] == 0), 'custom_' + feature.lower()] = 1
    train.loc[(train[feature] < down_) | (train[feature] > up_) & (train['Gender'] == 0), 'custom_' + feature.lower()] = 2
    train[custom] = train[custom].fillna(0)
    return train

def feature_engineer_bun(train):
    feature = 'BUN'
    up = 21
    down = 7
    custom = 'custom_' + feature.lower()
    train.loc[(train[feature] >= down) & (train[feature] <= up), 'custom_' + feature.lower()] = 1
    train.loc[(train[feature].between(up, 60)), 'custom_' + feature.lower()] = 2
    train.loc[(train[feature] < down), 'custom_' + feature.lower()] = 3
    train.loc[(train[feature] >= 60), 'custom_' + feature.lower()] = 4
    train.loc[(train['BUN'] != None) & (train['Creatinine'] != None) & ((train['BUN']/train['Creatinine'] >= 10) | (train['BUN']/train['Creatinine'] <= 20)), 'custom_BC_ratio'] = 1
    train[custom] = train[custom].fillna(0)
    train['custom_BC_ratio'] = train['custom_BC_ratio'].fillna(0)
    
    return train

def feature_engineer_creatinine(train):
    feature = 'Creatinine'
    up = 0.7
    down = 1.2
    up_ = 0.5
    down_ = 1.0
    custom = 'custom_' + feature.lower()
    train.loc[(train[feature] >= down) & (train[feature] <= up) & (train['Gender'] == 1), 'custom_' + feature.lower()] = 1
    train.loc[((train[feature] < down) | (train[feature] > up)) & (train['Gender'] == 1), 'custom_' + feature.lower()] = 2
    train.loc[(train[feature] >= down_) & (train[feature] <= up_) & (train['Gender'] == 0), 'custom_' + feature.lower()] = 1
    train.loc[((train[feature] < down_) | (train[feature] > up_)) & (train['Gender'] == 0), 'custom_' + feature.lower()] = 2
    train[custom] = train[custom].fillna(0)
    return train

def feature_engineer_alkalinephos(train):
    feature = 'Alkalinephos'
    up = 147
    down = 44
    custom = 'custom_' + feature.lower()
    train.loc[(train[feature] >= down) & (train[feature] <= up), 'custom_' + feature.lower()] = 1
    train.loc[(train[feature] < down) | (train[feature] > up), 'custom_' + feature.lower()] = 2
    train[custom] = train[custom].fillna(0)
    
    return train

def feature_engineer_calcium(train):
    feature = 'Calcium'
    up = 10.2
    down = 8.5
    custom = 'custom_' + feature.lower()
    train.loc[(train[feature] >= down) & (train[feature] <= up), 'custom_' + feature.lower()] = 1
    train.loc[(train[feature] < down) | (train[feature] > up), 'custom_' + feature.lower()] = 2
    train[custom] = train[custom].fillna(0)
    
    return train

def feature_engineer_chloride(train):
    feature = 'Chloride'
    up = 98
    down = 106
    custom = 'custom_' + feature.lower()
    train.loc[(train[feature] >= down) & (train[feature] <= up), 'custom_' + feature.lower()] = 1
    train.loc[((train[feature] >= 70) & (train[feature] < down)) | (train[feature] > up & (train[feature] <= 120)), 'custom_' + feature.lower()] = 2
    train.loc[(train[feature] < 70) | (train[feature] > 120), 'custom_' + feature.lower()] = 4
    
    train[custom] = train[custom].fillna(0)
    
    return train

def feature_engineer_bilirubin(train):
    feature = 'Bilirubin_direct'
    feature_ = 'Bilirubin_total'
    custom = 'custom_bd'
    custom_ = 'custom_bt'
    up = 0.3
    down = -1000
    up_ = 1.2
    down_ = 0.1
    
    train.loc[(train[feature] >= down) & (train[feature] <= up), custom] = 1
    train.loc[(train[feature] < down) | (train[feature] > up), custom] = 2
    
    train.loc[(train[feature_] >= down_) & (train[feature_] <= up_), custom_] = 1
    train.loc[(train[feature_] < down_) | (train[feature_] > up_), custom_] = 2
    
    train[custom] = train[custom].fillna(0)
    train[custom_] = train[custom_].fillna(0)
    return train

def feature_engineer_glucose(train):
    feature = 'Glucose'
    up = 125
    down = 100
    custom = 'custom_' + feature.lower()
    train.loc[(train[feature] >= down) & (train[feature] <= up), 'custom_' + feature.lower()] = 1
    train.loc[(train[feature] < down) | (train[feature].between(up, 200)), 'custom_' + feature.lower()] = 2
    train.loc[(train[feature] >= 200), 'custom_' + feature.lower()] = 4
    train[custom] = train[custom].fillna(0)
    
    return train

def feature_engineer_lactate(train):
    feature = 'Lactate'
    up = 18.2
    down = 4.55
    custom = 'custom_' + feature.lower()
    train.loc[(train[feature] >= down) & (train[feature] <= up), 'custom_' + feature.lower()] = 1
    train.loc[(train[feature] < down) | (train[feature] > up), 'custom_' + feature.lower()] = 2
    train[custom] = train[custom].fillna(0)
    
    return train

def feature_engineer_magnesium(train):
    feature = 'Magnesium'
    up = 1.1
    down = 0.6
    custom = 'custom_' + feature.lower()
    train.loc[(train[feature] >= down) & (train[feature] <= up), 'custom_' + feature.lower()] = 1
    train.loc[(train[feature].between(up,2.9)), 'custom_' + feature.lower()] = 2
    train.loc[(train[feature] < down), 'custom_' + feature.lower()] = 3
    train.loc[(train[feature] >= 2.9) , 'custom_' + feature.lower()] = 4
    train[custom] = train[custom].fillna(0)
    
    return train

def feature_engineer_phosphate(train):
    feature = 'Phosphate'
    up = 4.5
    down = 2.5
    custom = 'custom_' + feature.lower()
    train.loc[(train[feature] >= down) & (train[feature] <= up), 'custom_' + feature.lower()] = 1
    train.loc[(train[feature] < down) | (train[feature] > up), 'custom_' + feature.lower()] = 2
    train[custom] = train[custom].fillna(0)
    
    return train

def feature_engineer_potassium(train):
    feature = 'Potassium'
    up = 5.2
    down = 3.6
    custom = 'custom_' + feature.lower()
    train.loc[(train[feature] >= down) & (train[feature] <= up), 'custom_' + feature.lower()] = 1
    train.loc[(train[feature] < down) | (train[feature] > up) & (train[feature] <= 6), 'custom_' + feature.lower()] = 2
    train.loc[(train[feature] > 6), 'custom_' + feature.lower()] = 4
    train[custom] = train[custom].fillna(0)
    
    return train

def feature_engineer_troponin(train):
    feature = 'TroponinI'
    up = 0.4
    down = 0
    custom = 'custom_' + feature.lower()
    train.loc[(train[feature] >= down) & (train[feature] <= up), 'custom_' + feature.lower()] = 1
    train.loc[(train[feature] < down) | (train[feature] > up), 'custom_' + feature.lower()] = 2
    train[custom] = train[custom].fillna(0)
    
    return train

def feature_engineer_hct(train):
    feature = 'Hct'
    up = 52
    down = 45
    up_ = 48
    down_ = 37
    custom = 'custom_' + feature.lower()
    train.loc[(train[feature] >= down) & (train[feature] <= up) & (train['Gender'] == 1), 'custom_' + feature.lower()] = 1
    train.loc[((train[feature] < down) | (train[feature] > up)) & (train['Gender'] == 1), 'custom_' + feature.lower()] = 2
    train.loc[(train[feature] >= down_) & (train[feature] <= up_) & (train['Gender'] == 0), 'custom_' + feature.lower()] = 1
    train.loc[((train[feature] < down_) | (train[feature] > up_)) & (train['Gender'] == 0), 'custom_' + feature.lower()] = 2
    train[custom] = train[custom].fillna(0)
    
    return train

def feature_engineer_hgb(train):
    feature = 'Hgb'
    up = 17.5
    down = 13.5
    up_ = 15.5
    down_ = 12
    custom = 'custom_' + feature.lower()
    train.loc[(train[feature] >= down) & (train[feature] <= up) & (train['Gender'] == 1), 'custom_' + feature.lower()] = 1
    train.loc[((train[feature] < down) | (train[feature] > up)) & (train['Gender'] == 1), 'custom_' + feature.lower()] = 2
    train.loc[(train[feature] >= down_) & (train[feature] <= up_) & (train['Gender'] == 0), 'custom_' + feature.lower()] = 1
    train.loc[((train[feature] < down_) | (train[feature] > up_)) & (train['Gender'] == 0), 'custom_' + feature.lower()] = 2
    train[custom] = train[custom].fillna(0)
    return train

def feature_engineer_ptt(train):
    feature = 'PTT'
    up = 35
    down = 25
    custom = 'custom_' + feature.lower()
    train.loc[(train[feature] >= down) & (train[feature] <= up), 'custom_' + feature.lower()] = 1
    train.loc[(train[feature] < down) | (train[feature] > up), 'custom_' + feature.lower()] = 2
    train[custom] = train[custom].fillna(0)
    
    return train

def feature_engineer_wbc(train):
    feature = 'WBC'
    up = 11
    down = 4.5
    custom = 'custom_' + feature.lower()
    train.loc[(train[feature] >= down) & (train[feature] <= up), 'custom_' + feature.lower()] = 1
    train.loc[(train[feature] < down) | (train[feature] > up), 'custom_' + feature.lower()] = 2
    train[custom] = train[custom].fillna(0)
    
    return train

def feature_engineer_fibrinogen(train):
    feature = 'Fibrinogen'
    up = 400
    down = 150
    custom = 'custom_' + feature.lower()
    train.loc[(train[feature] >= down) & (train[feature] <= up), 'custom_' + feature.lower()] = 1
    train.loc[(train[feature] < down) | (train[feature] > up), 'custom_' + feature.lower()] = 2
    train[custom] = train[custom].fillna(0)
    
    return train

def feature_engineer_platelets(train):
    feature = 'Platelets'
    up = 400
    down = 150
    custom = 'custom_' + feature.lower()
    train.loc[(train[feature] >= down) & (train[feature] <= up), 'custom_' + feature.lower()] = 1
    train.loc[(train[feature] < down) | (train[feature] > up), 'custom_' + feature.lower()] = 2
    train[custom] = train[custom].fillna(0)
    
    return train

def feature_engineer_unit(train):
    
    train.loc[train['Unit1'] == 1, 'Unit'] = 1
    train.loc[train['Unit2'] == 1, 'Unit'] = 2
    train['Unit'] = train['Unit'].fillna(3)
    
    return train

def feature_engineer(train):
    train = feature_engineer_age(train)
    train = feature_engineer_blood_pressure(train)
    train = feature_engineer_hr(train)
    train = feature_engineer_o2sat(train)
    train = feature_engineer_temp(train)
    train = feature_engineer_resp_rate(train)
    train = feature_engineer_etco2(train)
    
    train = feature_engineer_baseexcess(train)
    train = feature_engineer_hco3(train)
    train = feature_engineer_fso2(train)
    train = feature_engineer_ph(train)
    train = feature_engineer_paco2(train)
    train = feature_engineer_ast(train)
    train = feature_engineer_bun(train)
    train = feature_engineer_alkalinephos(train)
    train = feature_engineer_calcium(train)
    train = feature_engineer_chloride(train)
    train = feature_engineer_creatinine(train)
    train = feature_engineer_bilirubin(train)
    train = feature_engineer_glucose(train)
    train = feature_engineer_lactate(train)
    train = feature_engineer_magnesium(train)
    train = feature_engineer_phosphate(train)
    train = feature_engineer_potassium(train)
    train = feature_engineer_troponin(train)
    train = feature_engineer_hct(train)
    train = feature_engineer_hgb(train)
    train = feature_engineer_ptt(train)
    train = feature_engineer_wbc(train)
    train = feature_engineer_fibrinogen(train)
    train = feature_engineer_platelets(train)
    train = feature_engineer_unit(train)
    train = feature_engineer_cut(train)
    
    return train

def w_binary_crossentropy(y_true, y_pred):
    return K.mean(tf.nn.weighted_cross_entropy_with_logits(
                                                           y_true,
                                                           y_pred,
                                                           1,
                                                           name=None
                                                           ), axis=-1)

def weighted_binary_crossentropy(weights):
    def w_binary_crossentropy(y_true, y_pred):
        return K.mean(tf.nn.weighted_cross_entropy_with_logits(
                                                               y_true,
                                                               y_pred,
                                                               weights,
                                                               name=None
                                                               ), axis=-1)
    return w_binary_crossentropy
kw_loss = weighted_binary_crossentropy(weights=1)

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


def feature_engineer_cut_np(train):
    tmp = train[-1:]
    while len(train) <= 100:
        train = np.concatenate((train,tmp),axis=0)
    if len(train) > 100:
        train = train[:100]
    return train

def fix_100(res, org_length):
    """
    input: 
            res label are given by predict
            org_length is the original length of each patient
            t is threshold
    output:
            fixed probabilities, original labels and predictions of original length,
            also processed by threshold
            
    """
    l = 100 # fixed length of each patient
    
    tmp_res = list(res)
    
    if org_length <= l:
        tmp_res = tmp_res[:org_length]
    else:
        last = tmp_res[-1]
        tmp_res += [last for _ in range(org_length- l)]

    tmp_predict = [1 for _ in range(len(tmp_res))]

    for r in range(len(tmp_res)):
        if tmp_res[r] < threshold:
            tmp_predict[r] = 0
        else:    
            r += 1
            while r < len(tmp_res):
                tmp_res[r] = tmp_res[r - 1]
                r += 1
            break

    return tmp_res, tmp_predict


def load_sepsis_model():
    
    #model = load_model(model_name) # you need this without self defined loss function
    model = load_model(model_name, custom_objects={'w_binary_crossentropy':w_binary_crossentropy,'fmeasure':fmeasure}) # only for self-defined weighted binary crossentropy situation
#    pkl_file = open(encoder_name, 'rb')
#    encoder = pickle.load(pkl_file)
#    pkl_file.close()

    return model

# change me everytime you change the model
NOW = '2019-07-31-17-42-55'
model_name = f'./model/LSTM_{NOW}.h5'
#encoder_name = f'./model/Encoder_{NOW}.pkl'
threshold = 0.8

def get_sepsis_score(data, model):
    
    LSTM_model = model
    #encoder = model[1]
    cur_train = pd.read_csv(data, sep='|')
    org_length = len(cur_train)
    cur_train = cur_train.fillna(method='pad')
    cur_train = feature_engineer(cur_train)[feature_to_use]
    kw_loss = weighted_binary_crossentropy(weights=1)
    #    for c in custom_:
#        cur_train[c][0:len(cur_train[c])] = encoder[c].transform(cur_train[c][0:len(cur_train[c])])

    
    preds = [0 for _ in range(len(cur_train))]
    org_pred = [0 for _ in range(len(cur_train))]
    tmp = [0 for _ in range(len(cur_train))]

    dtest = np.array(cur_train[custom])
    dtest = feature_engineer_cut_np(dtest)
    dtest = dtest.reshape(-1,100,len(custom))
    predicted = list(LSTM_model.predict(dtest)[0])
    org_label = cur_train['SepsisLabel']

    for t in range(len(cur_train)):
#        dtest = np.array(cur_train[custom][:t+1])
#        dtest = feature_engineer_cut_np(dtest)
#        dtest = dtest.reshape(-1,100,len(custom))
#        predicted_ = list(LSTM_model.predict(dtest)[0])
        preds[t] = predicted[t]
#        print(abs(predicted_[t] - predicted[t]) )
        #print(predicted_[:t+1] == predicted[:t+1])
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
                tmp_fp.append(-1)
                tmp_p.append(1-fp)
        tmp[t] = 0
        for i in range(len(tmp_p)):
            tmp[t] += (tmp_p[i] * tmp_fp[i])
# print(preds[t], tmp_fp,f'*----{t}--------')
# print(len(preds[t]))
        org_pred[t] = preds[t]
        if len(tmp_p) >= 2:
            tmp[t] /= len(tmp_p)
            preds[t] = org_pred[t] * 0.9 + 0.1 * tmp[t]
        else:
            preds[t] = org_pred[t]
                

                
    gap = max(preds) - 0 + 0.01
    preds = (preds / gap)
    preds = [p if p > 0 else 0 for p in preds]
#preds = [abs(x) for x in preds]
    #label = [1 if p >= threshold else 0 for p in preds]
    score, label = fix_100(preds, org_length)
    org_pred,_ = fix_100(org_pred, org_length)
    tmp,_ = fix_100(tmp, org_length)
    org_label, _ = fix_100(org_label, org_length)
#print(score)

#    org_label = [ol + 0.05 if ol == 1 else ol - 0.05 for ol in org_label]
#    plt.figure(1)
#    plt.scatter(list(range(len(org_label))), score,c = 'g',alpha = 0.1)
#    plt.scatter(list(range(len(org_label))), org_label,c = 'b',alpha = 0.1)
#    plt.scatter(list(range(len(org_label))), label,c = 'r',alpha = 0.1)
#    plt.scatter(list(range(len(org_label))), org_pred,c = 'black',alpha = 0.1)
#    plt.scatter(list(range(len(org_label))), tmp,c = 'y',alpha = 0.1)
#    plt.legend(['probability','true','prdicted','org_pred','top3'], loc = 'lower right')
#    plt.title('Predicted Result of N samples from set B by model trained by set A')
#    plt.show()

    return score, label, org_pred, tmp

