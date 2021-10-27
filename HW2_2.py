import numpy as np
import math
from numpy.core.fromnumeric import shape
import pandas as pd
import csv

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

    # x_train_nor = (x_train - x_train.mean()) / (x_train.std())
    # x_test_nor = (x_test - x_test.mean()) / (x_test.std())

    return x_train_nor, x_test_nor

def sigmoid(z):
    # beacuse the number is too small may cause overflow， use seterr to ignore
    old_settings = np.seterr(all='ignore')
    
    res = 1 / (1.0 + np.exp(-z))
    return np.clip(res, 1e-6, 1-1e-6)

def train(x_train, y_train):
    
    # 打亂data順序
    index = np.arange(x_train.shape[0])
    np.random.shuffle(index)
    x = x_train[index]
    y = y_train[index]

    # index of class 1 and class 2 
    idx1 = []
    idx2 = []
    idx = 0

    # cnt1 => number of class 1 、 cnt2 => number of class 2
    cnt1 = 0
    cnt2 = 0
    
    # mu = (1,106)
    mu1 = np.zeros((x_train.shape[1],))
    mu2 = np.zeros((x_train.shape[1],))
    
    #test 
    # mu11 = np.zeros((dim,))
    # mu22 = np.zeros((dim,))

    # covariance matrix = (106,106) 
    cov1 = np.full((x_train.shape[1],x_train.shape[1]), 0)
    cov2 = np.full((x_train.shape[1],x_train.shape[1]), 0)
    cov_share = np.full((x_train.shape[1],x_train.shape[1]),0)

    # TODO : try to implement probalistic generative model
    
    # sum of class 1 and class 2
    sum1 = np.zeros((x_train.shape[1],))
    sum2 = np.zeros((x_train.shape[1],))


    for y in y_train :
        if y == 1:
            cnt1 += 1
            idx1.append(idx)
        elif y == 0:
            cnt2 += 1
            idx2.append(idx)
        
        idx += 1

    for i in range(x_train.shape[1]):
        sum1[i] = np.sum(x_train[idx1,i])
        sum2[i] = np.sum(x_train[idx2,i])

    # calculate mean matrix 
    mu1 = sum1 / cnt1
    mu2 = sum2 / cnt2

    # calculate covariance matrix 
    cov1 = ( np.dot( ((x_train[idx1,:] - mu1)).transpose(), (x_train[idx1,:] - mu1) ) ) / (cnt1)
    cov2 = ( np.dot( ((x_train[idx2,:] - mu2)).transpose(), (x_train[idx2,:] - mu2) ) ) / (cnt2)
    cov_share = (cnt1 / (cnt1 + cnt2)) * cov1 + (cnt2 / (cnt1 + cnt2)) * cov2

    # for i in range(dim):
    #     mu1[i] = (np.sum(x_train[idx1,i]) / cnt1)
        
    #     if(mu11[i] == mu1[i] and len(idx1) == cnt1):
    #         print("1 is right")
    #     else:
    #         print("1 is wrong")
    
    # for i in range(dim):
    #     mu2[i] = (np.sum(x_train[idx2,i]) / cnt2)
        
    #     if(mu22[i] == mu2[i] and len(idx2) == cnt2):
    #         print("2 is right")
    #     else:
    #         print("2 is wrong")
    
    return mu1, mu2, cov_share, cnt1, cnt2

def predict(x_test, mu1, mu2, cov_share, N1, N2):
    inv_cov = np.linalg.inv(cov_share)

    # w = (1,106)
    w = np.dot((mu1 - mu2), inv_cov)
    b = (-0.5) * np.dot(np.dot(mu1.T, inv_cov), mu1) + (0.5) * np.dot(np.dot(mu2.T, inv_cov), mu2) + np.log(float(N1) / (N2))

    z = np.dot(w, x_test.T) + b
    pred = sigmoid(z)

    return pred

def testacc(y_train, y):
    # result = np.zeros((y.shape[0],))
    result = []

    for i in range(y.shape[0]):
        if y_train[i] == y[i] : 
            # result[i] = 1
            result.append(1)
        else :
            # result[i] = 0
            result.append(0)

    acc = np.sum(result) / len(result)

    return acc

if __name__ == "__main__":

    x_train, y_train, x_test = load_data()
    # print(x_train.shape)
    x_train, x_test = normalize(x_train, x_test)
    
    # feats = [0, 2, 3, 5, 10, 33, 41, 53]
    feats = [0, 2, 3, 4, 5, 10, 24, 25, 27, 29, 33, 41, 47, 53, 58]
    # feats = [0, 2, 3, 4, 5, 10, 24, 25, 27, 29, 33, 41, 47, 53, 58, 63]
    x_train = x_train[:,feats]
    x_test = x_test[:,feats]

    mu1, mu2, share_sigma, cnt1, cnt2 = train(x_train, y_train)

    # predict the model with test data
    y_result = predict(x_test, mu1, mu2, share_sigma, cnt1, cnt2)
    y_result = np.around(y_result)

    # write to csv
    # with open('predict.csv', 'w', newline = '') as csvf:
    #     # set up csv writer
    #     writer = csv.writer(csvf)
    #     writer.writerow(['ID', 'Label'])
    #     for i in range(int(len(y_result))):
    #         writer.writerow( [i + 1, int(y_result[i]) ] )


    y = predict(x_train, mu1, mu2, share_sigma, cnt1, cnt2)
    y = np.around(y) 

    result = testacc(y_train, y)

    print('Train acc = %f' % (result))

    