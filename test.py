from itertools import count
import keras
from keras.backend import maximum
import numpy as np
from numpy.core.numeric import normalize_axis_tuple
import pandas as pd
import csv
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
import keras.optimizers as opt
from tensorflow.python.ops.random_ops import categorical


dim = 106

def load_data():
    x_train = pd.read_csv('X_train')
    x_test = pd.read_csv('X_test')

    # x_train = x_train[feats]
    # x_test = x_test[feats]

    x_train = x_train.values
    x_test = x_test.values

    y_train = pd.read_csv('Y_train', header = None)
    y_train = y_train.values
    y_train = y_train.reshape(-1)

    return x_train, y_train, x_test

def normalize(x_train, x_test):
    # row vector
    x_all = np.concatenate((x_train, x_test), axis = 0)
    mean = np.mean(x_all, axis = 0)
    std = np.std(x_all, axis = 0)

    index = [0, 1, 3, 4, 5]
    mean_vec = np.zeros(x_all.shape[1])
    std_vec = np.ones(x_all.shape[1])
    mean_vec[index] = mean[index]
    std_vec[index] = std[index]

    x_all_nor = (x_all - mean_vec) / std_vec
    x_train_nor = x_all_nor[0:x_train.shape[0]]
    x_test_nor = x_all_nor[x_train.shape[0]:]

    return x_train_nor, x_test_nor

def train(x_train, y_train):

    # 打亂data順序
    index = np.arange(x_train.shape[0])
    np.random.shuffle(index)
    x = x_train[index]
    y = y_train[index]
    x_train = x_train.astype('float32')

    # set parameters
    batch = 50
    epoch = 50

    model = Sequential()
    model.add(Dense(units = 50, input_shape = (x_train.shape[0], x_train.shape[1]), activation = 'sigmoid'))
    for i in range(10):
        model.add(Dense(units = 50, activation = 'relu'))

    model.add(Dense(units = 1, activation = 'sigmoid'))
    # model.add(Dense(units = 1, activation = 'softplus'))
    # model.add(Dense(units = 1, activation = 'softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer = 'Adam', metrics = ['accuracy'])
    # model.compile(loss = 'binary_crossentropy', optimizer = 'adagrad', metrics = ['binary_accuracy'])
    # model.compile(loss = 'binary_crossentropy', optimizer = 'Adadelta', metrics = [keras.metrics.BinaryAccuracy(name="binary_accuracy", dtype=None, threshold=0.5)])
    model.fit(x_train, y_train, batch_size = batch, epochs = epoch)

    result = model.evaluate(x_train, y_train, batch_size = batch)
    print("Train accuracy is %f" %(result[1]))

    return model, batch

if __name__ == "__main__" :

    import matplotlib.pyplot as plt
    import seaborn as sns

    feats = []
    count = 0
    x = pd.read_csv('X_train')
    y = pd.read_csv('Y_train', header = None)
    x = x.values
    y = y.values
    y = y.reshape(-1)
    # y = y.values
    
    from keras.utils.np_utils import to_categorical
    categorical_label = to_categorical(y)
    print(y)

    # print(y.shape)
    # pd.set_option('display.max_rows', None)
    # 5 -> hours_per_week
    # a = x.corr()[x.columns[5]].sort_values(ascending = False)
    # b = x.corr()[x.columns[5]]
    # for i in range(106):
    #     if b[i] > 0.10 :
    #         feats.append(i)

    # print(feats)
    # plt.figure(figsize = (20,15))
    # sns.heatmap(x.corr(), cmap="jet", annot = False,linewidths = 2, robust = True)


    # find correleation with y
    # count = 0
    # feats = []
    # for i in range(106):
    #     y = pd.Series(y)
    #     t = x[:,i]
    #     t = pd.Series(t)
    #     result = t.corr(y)
    #     if(result > 0.02) :
    #         feats.append(i)
    #         count += 1
    # # print(count)
    # print(feats)
        # print(result)
   
    
    


    