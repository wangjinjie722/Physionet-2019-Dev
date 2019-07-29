import os
import sys
import copy
import time
import pickle
import random
import numpy as np
import pandas as pd
import xgboost as xgb
from tqdm import tqdm
from pylab import mpl
from os import listdir
import lightgbm as lgb
from keras import layers
from keras import models
from sklearn import metrics
from keras.layers import Dense
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.models import load_model
from sklearn.metrics import roc_auc_score
from keras.layers import BatchNormalization
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.layers import LSTM, Dropout, Bidirectional
from sklearn.preprocessing import LabelEncoder, OneHotEncoder,MinMaxScaler
from sklearn.metrics import confusion_matrix, precision_score, recall_score, \
    average_precision_score, precision_recall_curve


now = int(round(time.time()*1000))
NOW = time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(now/1000))

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

now = int(round(time.time()*1000))
NOW = time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(now/1000))
patient_length = 100
num_features = len(custom)

def sample_shuffle(train, label):
    """
        make the number of 01 samples the same and random shuffle their order
    """
    normal = []
    abnormal = []
    
    for i in range(len(train)):
        if sum(label[i]) == 0:
            normal.append(i)
        else:
            abnormal.append(i)

    random.shuffle(normal)
    normal = normal[:len(abnormal)] # sample to same for 1 and 0 training case
    normal = normal[:int(len(normal) * 1)]
    abnormal = abnormal[:int(len(abnormal) * 1)]
    index = normal + abnormal
    random.shuffle(index) # shuffle
    return train[index], label[index]


def build_model():
    """
        build model
    """
    model = models.Sequential()
    model.add(BatchNormalization())
    model.add(Bidirectional(LSTM(units=50,return_sequences=True), input_shape=(patient_length,num_features)))
    model.add(layers.Dense(200,activation='tanh'))
    model.add(Dropout(0.5))
    model.add(layers.Dense(patient_length,activation='sigmoid'))
    #model.compile(loss='binary_crossentropy', optimizer='adam')
    model.compile(loss='mse', optimizer='adam')
    return model

def data_process(meta_data):
    
    # get features and labels
    train_data = meta_data[custom]
    train_label = np.array(meta_data['SepsisLabel'])[:int(len(meta_data)//100) * 100]
    
    # encoding
    encoder = {}
    for c in tqdm(custom_):
        encoder[c] = LabelEncoder()
        tmp = encoder[c].fit_transform(train_data[c][0:len(train_data[c])])
        train_data[c][0:len(train_data[c])] = tmp

    # save encoder
    output = open(f'./model/Encoder_{NOW}.pkl', 'wb')
    pickle.dump(encoder, output)
    output.close()

    # cut train data
    train_data = np.array(train_data)[:int(len(meta_data)//100) * 100]

    #scaler = MinMaxScaler(feature_range=(0, 1))
    #train_data = scaler.fit_transform(train_data)
    train_label = np.asarray(train_label).astype('float32')

    train_size = len(meta_data)
    num_features = len(custom)
    patient_length = 100
    
    # resize to [samples, time steps, features]
    train_data = train_data.reshape(int(train_size/patient_length), patient_length, num_features)
    train_label = train_label.reshape(int(train_size/patient_length),patient_length)
    
    # sample and shuffle
    train_data_ss, train_label_ss  = sample_shuffle(train_data, train_label)
    
    # split
    val_size = int(len(train_data_ss) * 0.2)
    x_val = train_data_ss[len(train_data_ss) - val_size:]
    partial_x_train = train_data_ss[:len(train_data_ss) - val_size]
    y_val = train_label_ss[len(train_data_ss) - val_size:]
    partial_y_train = train_label_ss[:len(train_data_ss) - val_size]

    return partial_x_train, partial_y_train, x_val, y_val

if __name__ == '__main__':
    
    # load meta data
    with open('./data/cut.pkl', 'rb') as file:
        meta_data = pickle.load(file)
    
    # process data
    partial_x_train, partial_y_train, x_val, y_val = data_process(meta_data)
    
    # build model
    model = build_model()
    
    # begin to train!
    history = model.fit(partial_x_train,
                        partial_y_train,
                        epochs=50,
                        batch_size=24,
                        validation_data=(x_val,y_val))
                        
    model.save(f'./model/LSTM_{NOW}.h5')

    # plot
    plt.figure(1)
    plt.plot(history.history[ 'loss' ])
    plt.plot(history.history[ 'val_loss' ])
    plt.savefig(f"./result/history_{NOW}.png")


    with open('test_doc.txt', 'r') as fr:
        lines = fr.readlines()
    with open('test_doc.txt', 'w') as f:
        for line in lines:
            # 对每一行进行操作
            f.write(f'{line}') # 写入你想要的东西
        f.write(f'{NOW} | {sys.args[2]}')
