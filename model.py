from sklearn.utils import shuffle
import numpy as np
import pandas as pd
import csv
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


if __name__ == "__main__" :

    
    x_train, y_train, x_test = load_data()
    x_train, x_test = normalize(x_train, x_test)

    
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
    model1 = MLPClassifier(hidden_layer_sizes = 15, solver = "adam", shuffle = True, early_stopping = True, batch_size = 60)
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

    # # # write to csv
    with open('predict.csv', mode = 'w', newline = '') as csvf:
        writer = csv.writer(csvf)
        writer.writerow(['ID', 'Label'])
        for i in range(len(y_test)):
            writer.writerow( [i + 1, int(y_test[i]) ] )

    


    