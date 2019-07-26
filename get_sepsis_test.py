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


def load_sepsis_model():
    """
        \\
    """
    
    model = load_model(model_name)
    pkl_file = open(encoder_name, 'rb')
    encoder = pickle.load(pkl_file)
    pkl_file.close()

    return [model, encoder]

def get_sepsis_score(data, model):
    
    LSTM_model = model[0]
    encoder = model[1]
    cur_train = pd.read_csv(data, sep='|')
    org_length = len(cur_train)
    cur_train = cur_train.fillna(method='pad')
    cur_train = feature_engineer(cur_train)[feature_to_use]
    
    for c in custom_:
        #cur_train[c][0:len(cur_train[c])] = encoder_.fit_transform(cur_train[c][0:len(cur_train[c])])
        cur_train[c][0:len(cur_train[c])] = encoder[c].transform(cur_train[c][0:len(cur_train[c])])
    
    scaler = MinMaxScaler(feature_range=(0, 1))

    dtest = np.array(cur_train[custom])
    dtest = dtest.reshape(-1,100,len(custom))
    preds = LSTM_model.predict(dtest)
    
    #return preds, cur_train['SepsisLabel'], org_length

    org_label, = cur_train['SepsisLabel']
    Res = preds
    gap = max(Res) - 0 + 0.01
    Res = (Res / gap)
    Res = [abs(x) for x in Res]

    score, org_label, label = fix_100(Res, org_label, org_length, threshold)

    return score, label, org_label



if __name__ == '__main__':

    # inilization
    MODEL = load_sepsis_model()
    score = []
    org_label = []
    org_length = []

    pkl_file = open(sys.argv[1], 'rb')
    test_candidate = pickle.load(pkl_file)
    pkl_file.close()

    # test every case
    for i in (test_candidate):
        R,l,org_l = get_sepsis_score(i, MODEL)
    if len(res) == 0:
        res = list(R[0])
    else:
        res += list(R[0])
    org_label += list(l)
    org_length.append(org_l)

    # get results
    score, label, org_label = fix_100(res, label, org_length)

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

    # python3 evaluate_sepsis_score.py labels.zip predictions.zip




def fix_100(res, label, org_length):
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
    Res = []
    Label = []
    Predict = []
    for i in range(len(org_length)):
        tmp_res = res[i*l:(i+1)*l]
        tmp_label = label[i*l:(i+1)*l]
        if org_length[i] <= l:
            tmp_res = tmp_res[:org_length[i]]
            tmp_label = tmp_label[:org_length[i]]
        else:
            last = tmp_res[-1]
            tmp_res += [last for _ in range(org_length[i] - l)]
            tmp_label += [tmp_label[-1] for _ in range(org_length[i] - l)]
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
                
        Predict += tmp_predict
        Res += tmp_res
        Label += tmp_label

    return Res, Label, Predict 






def feature_engineer_cut(train):
    tmp = train.tail(1)
    while len(train) <= 100:
        train = pd.concat([train,tmp],ignore_index=True,)
    if len(train) > 100:
        train = train[:100]
    return train

def feature_engineer_hr(train):
    train.loc[(train['HR'] >= 100) & (train['Age'] >= 10 ),'custom_hr'] = 'abnormal'
    train.loc[(train['HR'] < 100) & (train['HR'] > 60) & (train['Age'] >= 10 ),'custom_hr'] = 'normal'
    train.loc[(train['HR'] >= 70) & (train['HR'] < 190) & (train['Age'] < 10 ),'custom_hr'] = 'normal'
    train.loc[((train['HR'] < 70) | (train['HR'] >= 190)) & (train['Age'] < 10 ),'custom_hr'] = 'abnormal'
    train['custom_hr'] = train['custom_hr'].fillna('Missing')
    return train

def feature_engineer_temp(train):
    train.loc[(train['Temp'] >= 36.4) & (train['Temp'] < 37.6), 'custom_temp'] = 'normal'
    train.loc[(train['Temp'] < 36.4) | (train['Temp'] >= 37.6), 'custom_temp'] = 'abnormal'
    train['custom_temp'] = train['custom_temp'].fillna('Missing')
    return train

def feature_engineer_age(train):
    train.loc[train['Age'] >=65, 'custom_age'] = 'old'
    train.loc[train['Age'] <10, 'custom_age'] = 'infant'
    train.loc[(train['Age'] >=10) & (train['Age'] <65), 'custom_age'] = 'adult'
    return train

def feature_engineer_o2sat(train):
    train.loc[(train['O2Sat'] >= 90) & (train['O2Sat'] < 100), 'custom_o2sat'] = 'normal'
    train.loc[(train['O2Sat'] < 90) & (train['O2Sat'] >= 0), 'custom_o2sat'] = 'abnormal'
    train['custom_o2sat'] = train['custom_o2sat'].fillna('Missing')
    return train

def feature_engineer_blood_pressure(train):
    train.loc[(train['SBP'] <90) & (train['DBP'] <60), 'custom_bp'] = 'low'
    train.loc[(train['SBP'].between(90,120, inclusive=True)) & (train['DBP'].between(60,80, inclusive=True)), 'custom_bp'] = 'normal'
    train.loc[(train['SBP'].between(120,140, inclusive=True)) & (train['DBP'].between(80,90, inclusive=True)), 'custom_bp'] = 'elevated'
    train.loc[(train['SBP'] > 140 ) & (train['DBP'] > 90 ), 'custom_bp'] = 'high'
    train['custom_bp'] = train['custom_bp'].fillna('Missing')
    return train

def feature_engineer_resp_rate(train):
    train.loc[(train['Resp'].between(30,60)) & (train['Age'] <1), 'custom_resp'] = 'normal'
    train.loc[((train['Resp'] < 30) | (train['Resp'] > 60)) & (train['Age'] <1) ,'custom_resp'] = 'abnormal'
    train.loc[(train['Resp'].between(24,40)) & (train['Age'].between(1,3)), 'custom_resp'] = 'normal'
    train.loc[((train['Resp'] < 24) | (train['Resp'] > 40)) & (train['Age'].between(1,3)) ,'custom_resp'] = 'abnormal'
    train.loc[(train['Resp'].between(22,34)) & (train['Age'].between(3,6)), 'custom_resp'] = 'normal'
    train.loc[((train['Resp'] < 22) | (train['Resp'] > 34)) & (train['Age'].between(3,6)) ,'custom_resp'] = 'abnormal'
    train.loc[(train['Resp'].between(18,30)) & (train['Age'].between(6,12)), 'custom_resp'] = 'normal'
    train.loc[((train['Resp'] < 18) | (train['Resp'] > 30)) & (train['Age'].between(6,12)) ,'custom_resp'] = 'abnormal'
    train.loc[(train['Resp'].between(12,20)) & (train['Age'] >12), 'custom_resp'] = 'normal'
    train.loc[((train['Resp'] < 12) | (train['Resp'] > 20)) & (train['Age'] >12),'custom_resp'] = 'abnormal'
    train['custom_resp'] = train['custom_resp'].fillna('Missing')
    return train

def feature_engineer_ph(train):
    feature = 'pH'
    custom = 'custom_ph'
    train.loc[(train[feature] >= 7.35) & (train[feature] <= 7.45), 'custom_ph'] = 'normal'
    train.loc[(train[feature] < 7.35) | (train[feature] > 7.45), 'custom_ph'] = 'abnormal'
    train[custom] = train[custom].fillna('Missing')
    return train

def feature_engineer_etco2(train):
    train.loc[(train['EtCO2'] >= 35) & (train['EtCO2'] <=45), 'custom_etco2'] = 'normal'
    train.loc[(train['EtCO2'] < 35) | (train['EtCO2'] > 45), 'custom_etco2'] = 'abnormal'
    train['custom_etco2'] = train['custom_etco2'].fillna('Missing')
    return train

def feature_engineer_baseexcess(train):
    feature = 'BaseExcess'
    train.loc[(train['BaseExcess'] >= -3) & (train['BaseExcess'] <= 3), 'custom_be'] = 'normal'
    train.loc[(train['BaseExcess'] < -3) | (train['BaseExcess'] > 3), 'custom_be'] = 'abnormal'
    train['custom_be'] = train['custom_be'].fillna('Missing')
    return train

def feature_engineer_hco3(train):
    feature = 'HCO3'
    train.loc[(train[feature] >= 22) & (train[feature] <= 27), 'custom_hco3'] = 'normal'
    train.loc[(train[feature] < 22) | (train[feature] > 27), 'custom_hco3'] = 'abnormal'
    train['custom_hco3'] = train['custom_hco3'].fillna('Missing')
    return train

def feature_engineer_fso2(train):
    feature = 'FiO2'
    feature_ = 'SaO2'
    custom = 'custom_fio2'
    custom_ = 'custom_sao2'
    
    train.loc[(train[feature] >= 0.18) & (train[feature] <= 0.25), custom] = 'normal'
    train.loc[(train[feature] < 0.18) | (train[feature] > 0.25), custom] = 'abnormal'
    
    train.loc[(train[feature_] >= 94) & (train[feature_] <= 100), custom_] = 'normal'
    train.loc[(train[feature_] < 94) | (train[feature_] > 100), custom_] = 'abnormal'
    
    train[custom] = train[custom].fillna('Missing')
    train[custom_] = train[custom_].fillna('Missing')
    return train

def feature_engineer_paco2(train):
    feature = 'PaCO2'
    custom = 'custom_paco2'
    train.loc[(train[feature] >= 38) & (train[feature] <= 42), 'custom_paco2'] = 'normal'
    train.loc[(train[feature].between(42, 59)), 'custom_paco2'] = 'moderate'
    train.loc[(train[feature] >= 60), 'custom_paco2'] = 'serious'
    train.loc[(train[feature] < 38), 'custom_paco2'] = 'low'
    
    train[custom] = train[custom].fillna('Missing')
    return train

def feature_engineer_ast(train):
    feature = 'AST'
    up = 34
    down = 6
    up_ = 40
    down_ = 8
    custom = 'custom_' + feature.lower()
    train.loc[(train[feature] >= down) & (train[feature] <= up) & (train['Gender'] == 1), 'custom_' + feature.lower()] = 'normal'
    train.loc[(train[feature] < down) | (train[feature] > up) & (train['Gender'] == 1), 'custom_' + feature.lower()] = 'abnormal'
    train.loc[(train[feature] >= down_) & (train[feature] <= up_) & (train['Gender'] == 0), 'custom_' + feature.lower()] = 'normal'
    train.loc[(train[feature] < down_) | (train[feature] > up_) & (train['Gender'] == 0), 'custom_' + feature.lower()] = 'abnormal'
    train[custom] = train[custom].fillna('Missing')
    return train

def feature_engineer_bun(train):
    feature = 'BUN'
    up = 21
    down = 7
    custom = 'custom_' + feature.lower()
    train.loc[(train[feature] >= down) & (train[feature] <= up), 'custom_' + feature.lower()] = 'normal'
    train.loc[(train[feature].between(up, 60)), 'custom_' + feature.lower()] = 'abnormal'
    train.loc[(train[feature] < down), 'custom_' + feature.lower()] = 'low'
    train.loc[(train[feature] >= 60), 'custom_' + feature.lower()] = 'serious'
    train.loc[(train['BUN'] != None) & (train['Creatinine'] != None) & ((train['BUN']/train['Creatinine'] >= 10) | (train['BUN']/train['Creatinine'] <= 20)), 'custom_BC_ratio'] = 'normal'
    train[custom] = train[custom].fillna('Missing')
    train['custom_BC_ratio'] = train['custom_BC_ratio'].fillna('Missing')
    
    return train

def feature_engineer_creatinine(train):
    feature = 'Creatinine'
    up = 0.7
    down = 1.2
    up_ = 0.5
    down_ = 1.0
    custom = 'custom_' + feature.lower()
    train.loc[(train[feature] >= down) & (train[feature] <= up) & (train['Gender'] == 1), 'custom_' + feature.lower()] = 'normal'
    train.loc[((train[feature] < down) | (train[feature] > up)) & (train['Gender'] == 1), 'custom_' + feature.lower()] = 'abnormal'
    train.loc[(train[feature] >= down_) & (train[feature] <= up_) & (train['Gender'] == 0), 'custom_' + feature.lower()] = 'normal'
    train.loc[((train[feature] < down_) | (train[feature] > up_)) & (train['Gender'] == 0), 'custom_' + feature.lower()] = 'abnormal'
    train[custom] = train[custom].fillna('Missing')
    return train

def feature_engineer_alkalinephos(train):
    feature = 'Alkalinephos'
    up = 147
    down = 44
    custom = 'custom_' + feature.lower()
    train.loc[(train[feature] >= down) & (train[feature] <= up), 'custom_' + feature.lower()] = 'normal'
    train.loc[(train[feature] < down) | (train[feature] > up), 'custom_' + feature.lower()] = 'abnormal'
    train[custom] = train[custom].fillna('Missing')
    
    return train

def feature_engineer_calcium(train):
    feature = 'Calcium'
    up = 10.2
    down = 8.5
    custom = 'custom_' + feature.lower()
    train.loc[(train[feature] >= down) & (train[feature] <= up), 'custom_' + feature.lower()] = 'normal'
    train.loc[(train[feature] < down) | (train[feature] > up), 'custom_' + feature.lower()] = 'abnormal'
    train[custom] = train[custom].fillna('Missing')
    
    return train

def feature_engineer_chloride(train):
    feature = 'Chloride'
    up = 98
    down = 106
    custom = 'custom_' + feature.lower()
    train.loc[(train[feature] >= down) & (train[feature] <= up), 'custom_' + feature.lower()] = 'normal'
    train.loc[((train[feature] >= 70) & (train[feature] < down)) | (train[feature] > up & (train[feature] <= 120)), 'custom_' + feature.lower()] = 'abnormal'
    train.loc[(train[feature] < 70) | (train[feature] > 120), 'custom_' + feature.lower()] = 'serious'
    
    train[custom] = train[custom].fillna('Missing')
    
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
    
    train.loc[(train[feature] >= down) & (train[feature] <= up), custom] = 'normal'
    train.loc[(train[feature] < down) | (train[feature] > up), custom] = 'abnormal'
    
    train.loc[(train[feature_] >= down_) & (train[feature_] <= up_), custom_] = 'normal'
    train.loc[(train[feature_] < down_) | (train[feature_] > up_), custom_] = 'abnormal'
    
    train[custom] = train[custom].fillna('Missing')
    train[custom_] = train[custom_].fillna('Missing')
    return train

def feature_engineer_glucose(train):
    feature = 'Glucose'
    up = 125
    down = 100
    custom = 'custom_' + feature.lower()
    train.loc[(train[feature] >= down) & (train[feature] <= up), 'custom_' + feature.lower()] = 'normal'
    train.loc[(train[feature] < down) | (train[feature].between(up, 200)), 'custom_' + feature.lower()] = 'abnormal'
    train.loc[(train[feature] >= 200), 'custom_' + feature.lower()] = 'serious'
    train[custom] = train[custom].fillna('Missing')
    
    return train

def feature_engineer_lactate(train):
    feature = 'Lactate'
    up = 18.2
    down = 4.55
    custom = 'custom_' + feature.lower()
    train.loc[(train[feature] >= down) & (train[feature] <= up), 'custom_' + feature.lower()] = 'normal'
    train.loc[(train[feature] < down) | (train[feature] > up), 'custom_' + feature.lower()] = 'abnormal'
    train[custom] = train[custom].fillna('Missing')
    
    return train

def feature_engineer_magnesium(train):
    feature = 'Magnesium'
    up = 1.1
    down = 0.6
    custom = 'custom_' + feature.lower()
    train.loc[(train[feature] >= down) & (train[feature] <= up), 'custom_' + feature.lower()] = 'normal'
    train.loc[(train[feature].between(up,2.9)), 'custom_' + feature.lower()] = 'abnormal'
    train.loc[(train[feature] < down), 'custom_' + feature.lower()] = 'low'
    train.loc[(train[feature] >= 2.9) , 'custom_' + feature.lower()] = 'serious'
    train[custom] = train[custom].fillna('Missing')
    
    return train

def feature_engineer_phosphate(train):
    feature = 'Phosphate'
    up = 4.5
    down = 2.5
    custom = 'custom_' + feature.lower()
    train.loc[(train[feature] >= down) & (train[feature] <= up), 'custom_' + feature.lower()] = 'normal'
    train.loc[(train[feature] < down) | (train[feature] > up), 'custom_' + feature.lower()] = 'abnormal'
    train[custom] = train[custom].fillna('Missing')
    
    return train

def feature_engineer_potassium(train):
    feature = 'Potassium'
    up = 5.2
    down = 3.6
    custom = 'custom_' + feature.lower()
    train.loc[(train[feature] >= down) & (train[feature] <= up), 'custom_' + feature.lower()] = 'normal'
    train.loc[(train[feature] < down) | (train[feature] > up) & (train[feature] <= 6), 'custom_' + feature.lower()] = 'abnormal'
    train.loc[(train[feature] > 6), 'custom_' + feature.lower()] = 'serious'
    train[custom] = train[custom].fillna('Missing')
    
    return train

def feature_engineer_troponin(train):
    feature = 'TroponinI'
    up = 0.4
    down = 0
    custom = 'custom_' + feature.lower()
    train.loc[(train[feature] >= down) & (train[feature] <= up), 'custom_' + feature.lower()] = 'normal'
    train.loc[(train[feature] < down) | (train[feature] > up), 'custom_' + feature.lower()] = 'abnormal'
    train[custom] = train[custom].fillna('Missing')
    
    return train

def feature_engineer_hct(train):
    feature = 'Hct'
    up = 52
    down = 45
    up_ = 48
    down_ = 37
    custom = 'custom_' + feature.lower()
    train.loc[(train[feature] >= down) & (train[feature] <= up) & (train['Gender'] == 1), 'custom_' + feature.lower()] = 'normal'
    train.loc[((train[feature] < down) | (train[feature] > up)) & (train['Gender'] == 1), 'custom_' + feature.lower()] = 'abnormal'
    train.loc[(train[feature] >= down_) & (train[feature] <= up_) & (train['Gender'] == 0), 'custom_' + feature.lower()] = 'normal'
    train.loc[((train[feature] < down_) | (train[feature] > up_)) & (train['Gender'] == 0), 'custom_' + feature.lower()] = 'abnormal'
    train[custom] = train[custom].fillna('Missing')
    
    return train

def feature_engineer_hgb(train):
    feature = 'Hgb'
    up = 17.5
    down = 13.5
    up_ = 15.5
    down_ = 12
    custom = 'custom_' + feature.lower()
    train.loc[(train[feature] >= down) & (train[feature] <= up) & (train['Gender'] == 1), 'custom_' + feature.lower()] = 'normal'
    train.loc[((train[feature] < down) | (train[feature] > up)) & (train['Gender'] == 1), 'custom_' + feature.lower()] = 'abnormal'
    train.loc[(train[feature] >= down_) & (train[feature] <= up_) & (train['Gender'] == 0), 'custom_' + feature.lower()] = 'normal'
    train.loc[((train[feature] < down_) | (train[feature] > up_)) & (train['Gender'] == 0), 'custom_' + feature.lower()] = 'abnormal'
    train[custom] = train[custom].fillna('Missing')
    return train

def feature_engineer_ptt(train):
    feature = 'PTT'
    up = 35
    down = 25
    custom = 'custom_' + feature.lower()
    train.loc[(train[feature] >= down) & (train[feature] <= up), 'custom_' + feature.lower()] = 'normal'
    train.loc[(train[feature] < down) | (train[feature] > up), 'custom_' + feature.lower()] = 'abnormal'
    train[custom] = train[custom].fillna('Missing')
    
    return train

def feature_engineer_wbc(train):
    feature = 'WBC'
    up = 11
    down = 4.5
    custom = 'custom_' + feature.lower()
    train.loc[(train[feature] >= down) & (train[feature] <= up), 'custom_' + feature.lower()] = 'normal'
    train.loc[(train[feature] < down) | (train[feature] > up), 'custom_' + feature.lower()] = 'abnormal'
    train[custom] = train[custom].fillna('Missing')
    
    return train

def feature_engineer_fibrinogen(train):
    feature = 'Fibrinogen'
    up = 400
    down = 150
    custom = 'custom_' + feature.lower()
    train.loc[(train[feature] >= down) & (train[feature] <= up), 'custom_' + feature.lower()] = 'normal'
    train.loc[(train[feature] < down) | (train[feature] > up), 'custom_' + feature.lower()] = 'abnormal'
    train[custom] = train[custom].fillna('Missing')
    
    return train

def feature_engineer_platelets(train):
    feature = 'Platelets'
    up = 400
    down = 150
    custom = 'custom_' + feature.lower()
    train.loc[(train[feature] >= down) & (train[feature] <= up), 'custom_' + feature.lower()] = 'normal'
    train.loc[(train[feature] < down) | (train[feature] > up), 'custom_' + feature.lower()] = 'abnormal'
    train[custom] = train[custom].fillna('Missing')
    
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
