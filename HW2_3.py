from re import X
from keras import metrics
from keras import callbacks
from keras.backend import learning_phase, print_tensor, shape
from sklearn.utils import shuffle
import tensorflow as tf
import keras 
import numpy as np
import pandas as pd
import csv
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.models import Model
from keras import optimizers
from keras.callbacks import EarlyStopping
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings("ignore")



dim = 106

def load_data():
    x_train = pd.read_csv('X_train')
    x_test = pd.read_csv('X_test')

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

    # x_train = x_train.astype('float32')
    
    # set parameters
    batch = 100
    epoch = 30
    # learning_rate = 0.001
    # Op = tf.compat.v1.keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=10e-8, decay = 1e-4)

    model = keras.Sequential()
    model.add(Dense(units = 10, input_shape = (x_train.shape[0], x_train.shape[1]), activation = 'sigmoid'))
    for i in range(epoch):
        model.add(Dense(units = 10, activation = 'sigmoid'))
    
    model.add(Dense(units = 1, activation = 'sigmoid'))
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['binary_accuracy'])
    
    model.fit(x_train, y_train, batch_size = batch, epochs = epoch, shuffle = True)

    result = model.evaluate(x_train, y_train, batch_size = batch)
    print("Train accuracy is %f" %(result[1]))
    print("loss is %f" %(result[0]))

    # return model, batch
    return 1,2

if __name__ == "__main__" :

    
    x_train, y_train, x_test = load_data()
    x_train, x_test = normalize(x_train, x_test)
    # feats = [0, 2, 3, 5, 10, 33, 41, 53]
    # x_train = x_train[:,feats]
    # x_test = x_test[:,feats]

    # K-fold Cross Validation model evaluation
    # fold_no = 8
    # # # Define the K-fold Cross Validator
    # kfold = KFold(n_splits = fold_no, shuffle = True)
    
    # keras
    # batch = 50
    # epoch = 10
    # Op = tf.compat.v1.keras.optimizers.Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999, decay = 0)

    # model = tf.keras.Sequential()
    # model.add(Dense(units = 17, activation = 'relu', input_shape = (x_train.shape[1],)))

    # for i in range(16):
        # model.add(Dense(units = (16-i), activation = 'relu')) 
        # model.add(EarlyStopping(monitor = 'val_loss'))
    # # model.add(BatchNormalization())
    # model.add(Dense(units = 2, activation = 'relu')) 
    # # model.add(BatchNormalization())
    # model.add(Dense(units = 1, activation = 'relu')) 
    # # model.add(BatchNormalization())

    # model.add(Dense(units = 1, activation = 'relu'))
    # model.add(BatchNormalization())
    # model.compile(loss = 'binary_crossentropy', optimizer = Op, metrics = 'accuracy')
    # model.fit(x_train, y_train, batch_size = batch, epochs = epoch, validation_split = 0.1, shuffle = True, verbose = 2, callbacks = EarlyStopping())

    # result = model.evaluate(x_train, y_train)
    # print("Train accuracy is %f" %(result[1]))
    # print("loss is %f" %(result[0]))
    # optim = tf.keras.optimizers.Adam(learing_rate = 0.001)
    # for train, test in kfold.split(x_train, y_train):
        
    #     model = tf.keras.Sequential()
    #     model.add(Dense(units = 10, activation = 'sigmoid'))

    #     for i in range(3):
    #         model.add(Dense(units = 10, activation = 'relu')) 
    #         model.add(BatchNormalization())


    #     model.add(Dense(units = 1, activation = 'sigmoid'))
    #     model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['binary_accuracy'])
    #     model.fit(x_train[train], y_train[train], batch_size = batch, epochs = epoch, validation_split = 0.3, shuffle = True)

    #     result = model.evaluate(x_train[test], y_train[test], batch_size = batch)
    #     print("Train accuracy is %f" %(result[1]))
    # x_test = x_test.astype('float64')
    # y_test = model.predict(x_test)
    
    # sklearn model
    # from sklearn.model_selection import train_test_split
    # X1, X2, y1, y2 = train_test_split(x_train, y_train, random_state = 1, train_size=0.9, test_size=0.1)
    # model1 = LogisticRegression(random_state = 0)
    # model1 = model1.fit(X1,y1)
    # pred1 = model1.predict(X2)
    # logistic
    # model1 = LogisticRegression(random_state = 1)
    # model1 = model1.fit(x_train, y_train)
    # pred1 = model1.predict(x_train)
    
    # MLPClassifier
    from sklearn.neural_network import MLPClassifier
    model1 = MLPClassifier(hidden_layer_sizes = 15, solver = "adam", shuffle = True, early_stopping = True, batch_size = 50)
    model1 = model1.fit(x_train, y_train)
    pred1 = model1.predict(x_train)
    # model1 = MLPClassifier(validation_fraction = 0.1,hidden_layer_sizes = 10, solver = "adam", shuffle = False, early_stopping = True, batch_size = 50)
    # model1 = model1.fit(x_train, y_train)
    # pred1 = model1.predict(x_train)

    # decision tree
    # from sklearn.tree import DecisionTreeClassifier
    # model1 = DecisionTreeClassifier(random_state = 0)
    # model1 = model1.fit(X1,y1)
    # pred1 = model1.predict(X2)


    # from sklearn.metrics import accuracy_score
    # acc = accuracy_score(y_train, pred1) 
    # print(acc)
    
    y_test = model1.predict(x_test)
    


    count = 0

    for num in y_test:
        if num == 1 :
            count += 1
    
    print(count)

    # # print(y_test.shape)
    # # # write to csv
    with open('predict.csv', mode = 'w', newline = '') as csvf:
        writer = csv.writer(csvf)
        writer.writerow(['ID', 'Label'])
        for i in range(len(y_test)):
            writer.writerow( [i + 1, int(y_test[i]) ] )

    


    