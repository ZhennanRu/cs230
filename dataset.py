# create an interface function to get splited dataset
# normalization could be added later

import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import random

# train, valid, test data are all in list consistes of [mfccs, file name]
# want to change the list to two outputs:
# X: vector of mfccs
# Y: vector of labels
def data2vector(data):
    X = []
    Y = []
    for i in range(len(data)):
        X.append(data[i][0])
        Y.append(data[i][1])
    X = np.array(X)
    Y = np.array(Y)
    return X,Y

def getMFCCDataset(train_size = 0.6, valid_size = 0.2, test_size = 0.2):

    MFCCs_DATA = "MFCCsData"
    numpy_datas = []

    dirlist = os.listdir(MFCCs_DATA)
    for d in dirlist:
        d = os.path.join(MFCCs_DATA, d)
        datalist = os.listdir(d)
        datalist = [[np.load(os.path.join(d,x)), os.path.join(d,x)] for x in datalist]
        numpy_datas.extend(datalist)

    for i in range(len(numpy_datas)):
        numpy_datas[i][0] = np.transpose(np.resize(numpy_datas[i][0], (19,512)))
    collection = {}
    angry = []
    happy = []
    normal = []

    for i in range(len(numpy_datas)):
        file_name = numpy_datas[i][1]
        if "angry" in file_name:
            numpy_datas[i][1] = np.array([1,0,0])
            angry.append(numpy_datas[i])
        elif "happy" in file_name:
            numpy_datas[i][1] = np.array([0,1,0])
            happy.append(numpy_datas[i])
        else:
            numpy_datas[i][1] = np.array([0,0,1])
            normal.append(numpy_datas[i])

    random.shuffle(angry)
    random.shuffle(happy)
    random.shuffle(normal)

    train_data = angry[:int(len(angry)*train_size)] + happy[:int(len(happy)*train_size)] + normal[:int(len(normal)*train_size)]
    valid_data = angry[int(len(angry)*train_size):int(len(angry)*(train_size+valid_size))] + happy[int(len(happy)*train_size):int(len(happy)*(train_size+valid_size))] + normal[int(len(normal)*train_size):int(len(normal)*(train_size+valid_size))]
    test_data = angry[int(len(angry)*(train_size+valid_size)):] + happy[int(len(happy)*(train_size+valid_size)):] + normal[int(len(normal)*(train_size+valid_size)):]
    random.shuffle(train_data)
    random.shuffle(valid_data)
    random.shuffle(test_data)

    X_train, Y_train = data2vector(train_data)
    X_valid, Y_valid = data2vector(valid_data)
    X_test, Y_test = data2vector(test_data)

    collection['X_train'] = X_train
    collection['Y_train'] = Y_train
    collection['X_valid'] = X_valid
    collection['Y_valid'] = Y_valid
    collection['X_test'] = X_test
    collection['Y_test'] = Y_test
    
    X_train_miu = np.average(X_train, axis=0)
    X_train_std = np.std(X_train, axis=0)

    X_train_norm = (X_train - X_train_miu)/X_train_std
    X_valid_norm = (X_valid - X_train_miu)/X_train_std
    X_test_norm = (X_test - X_train_miu)/X_train_std

    collection['X_train_norm'] = X_train_norm
    collection['X_valid_norm'] = X_valid_norm
    collection['X_test_norm'] = X_test_norm

    # further add normalization
    return collection