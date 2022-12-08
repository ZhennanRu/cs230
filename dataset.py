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

# get Japanese dataset
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
        numpy_datas[i][0] = np.pad(numpy_datas[i][0], ((0, 0),(0, 1841-numpy_datas[i][0].shape[1])), 'constant', constant_values=(0,0))
    
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

# get RAVDESS dataset
def getMFCCDatasetRAVDESS(train_size = 0.6, valid_size = 0.2, test_size = 0.2):

    MFCCs_DATA = "MFCCsData_RAVDESS"
    numpy_datas = []

    dirlist = os.listdir(MFCCs_DATA)
    for d in dirlist:
        d = os.path.join(MFCCs_DATA, d)
        datalist = os.listdir(d)
        datalist = [[np.load(os.path.join(d,x)), os.path.join(d,x)] for x in datalist]
        numpy_datas.extend(datalist)

    for i in range(len(numpy_datas)):
        numpy_datas[i][0] = np.transpose(np.resize(numpy_datas[i][0], (19,1500)))

        
    collection = {}
    # change file name to correct label
    angry = []
    happy = []
    normal = []
    calm = []
    sad = []
    fearful = []
    disgust = []
    surprised = []

    for i in range(len(numpy_datas)):
        file_name = numpy_datas[i][1]
        if file_name[34] == '5':
            numpy_datas[i][1] = np.array([0,0,0,0,1,0,0,0])
            angry.append(numpy_datas[i])
        elif file_name[34] == '3':
            numpy_datas[i][1] = np.array([0,0,1,0,0,0,0,0])
            happy.append(numpy_datas[i])
        elif file_name[34] == '1':
            numpy_datas[i][1] = np.array([1,0,0,0,0,0,0,0])
            normal.append(numpy_datas[i])
        elif file_name[34] == '2':
            numpy_datas[i][1] = np.array([0,1,0,0,0,0,0,0])
            calm.append(numpy_datas[i])
        elif file_name[34] == '4':
            numpy_datas[i][1] = np.array([0,0,0,1,0,0,0,0])
            sad.append(numpy_datas[i])
        elif file_name[34] == '6':
            numpy_datas[i][1] = np.array([0,0,0,0,0,1,0,0])
            fearful.append(numpy_datas[i])
        elif file_name[34] == '7':
            numpy_datas[i][1] = np.array([0,0,0,0,0,0,1,0])
            disgust.append(numpy_datas[i])
        elif file_name[34] == '8':
            numpy_datas[i][1] = np.array([0,0,0,0,0,0,0,1])
            surprised.append(numpy_datas[i])

    random.shuffle(angry)
    random.shuffle(happy)
    random.shuffle(normal)
    random.shuffle(calm)
    random.shuffle(sad)
    random.shuffle(fearful)
    random.shuffle(disgust)
    random.shuffle(surprised)

    train_data = (
        angry[:int(len(angry)*train_size)] + 
        happy[:int(len(happy)*train_size)] + 
        normal[:int(len(normal)*train_size)] + 
        calm[:int(len(calm)*train_size)] + 
        sad[:int(len(sad)*train_size)] + 
        fearful[:int(len(fearful)*train_size)] + 
        disgust[:int(len(disgust)*train_size)] + 
        surprised[:int(len(surprised)*train_size)]
    )

    valid_data = (
        angry[int(len(angry)*train_size):int(len(angry)*(train_size+valid_size))] + 
        happy[int(len(happy)*train_size):int(len(happy)*(train_size+valid_size))] + 
        normal[int(len(normal)*train_size):int(len(normal)*(train_size+valid_size))] + 
        sad[int(len(sad)*train_size):int(len(sad)*(train_size+valid_size))] + 
        calm[int(len(calm)*train_size):int(len(calm)*(train_size+valid_size))] + 
        fearful[int(len(fearful)*train_size):int(len(fearful)*(train_size+valid_size))] + 
        disgust[int(len(disgust)*train_size):int(len(disgust)*(train_size+valid_size))] + 
        surprised[int(len(surprised)*train_size):int(len(surprised)*(train_size+valid_size))]
    )

    test_data = (
        angry[int(len(angry)*(train_size+valid_size)):] + 
        happy[int(len(happy)*(train_size+valid_size)):] + 
        normal[int(len(normal)*(train_size+valid_size)):] +  
        calm[int(len(calm)*(train_size+valid_size)):] + 
        sad[int(len(sad)*(train_size+valid_size)):] + 
        fearful[int(len(fearful)*(train_size+valid_size)):] + 
        disgust[int(len(disgust)*(train_size+valid_size)):] + 
        surprised[int(len(surprised)*(train_size+valid_size)):]
    )

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