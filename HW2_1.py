import numpy as np
import pandas as pd
import csv

# logistic regression

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

def sigmoid(z):
    res = 1 / (1.0 + np.exp(-z))
    # 1e-6 <= res <= 1-1e-6
    return np.clip(res, 1e-6, 1-1e-6)

def train(x_train, y_train):
    
    # 打亂data順序
    index = np.arange(x_train.shape[0])
    np.random.shuffle(index)
    x = x_train[index]
    y = y_train[index]

    batch_size = 100

    # w = (1,106) ，y = (1,32561) ， x = (32561,106)
    # w,b is for class of < 50k
    b = 0.0
    w = np.zeros(x_train.shape[1])
    lrate = 0.001
    epoch = 256
    # w_lr,b_lr is for class of > 50k
    b_lr = 0
    w_lr = np.ones(x_train.shape[1])
    epilson = 1e-8
    # class 1
    gt_w_sum_1 = 0
    gt_b_sum_1 = 0
    gt_w_tot_1 = 0
    gt_b_tot_1 = 0
    # class 2
    gt_w_sum_2 = 0
    gt_b_sum_2 = 0
    gt_w_tot_2 = 0
    gt_b_tot_2 = 0

    count = 0

    for e in range(epoch):
        for batch in range(int(x_train.shape[0]/batch_size)):        
            # TODO : try to implement gradient descent

            count += 1

            x_batch = x_train[batch*batch_size:(batch+1)*batch_size]
            y_batch = y_train[batch*batch_size:(batch+1)*batch_size]
            z1 = np.dot(w,x_batch.transpose()) + b
            z2 = np.dot(w_lr,x_batch.transpose()) + b_lr
            f1 = sigmoid(z1)
            f2 = sigmoid(z2)
            # loss = (1,32561)
            loss1 = y_batch - f1
            loss2 = y_batch - f2
            # calculate weight
            # class 1
            # gt_w = (106,1)
            gt_w_1 = np.dot(x_batch.transpose(),loss1.transpose()) * (-1)
            gt_w_sum_1 += gt_w_1 ** 2 
            gt_w_sum_avg_1 = gt_w_sum_1 / count     
            gt_w_tot_1 = (gt_w_sum_avg_1 + epilson) ** (-0.5)
            # class 2
            gt_w_2 = np.dot(x_batch.transpose(),loss2.transpose()) * (-1)
            gt_w_sum_2 += gt_w_2 ** 2 
            gt_w_sum_avg_2 = gt_w_sum_2 / count     
            gt_w_tot_2 = (gt_w_sum_avg_2 + epilson) ** (-0.5)

            # calculate bias
            # class 1
            gt_b_1 = (loss1.transpose().sum(axis = 0)) * (-1)
            gt_b_sum_1 += gt_b_1 ** 2
            gt_b_sum_avg_1 = gt_b_sum_1 / count
            gt_b_tot_1 = (gt_b_sum_avg_1 + epilson) ** (-0.5)
            # class 2
            gt_b_2 = (loss2.transpose().sum(axis = 0)) * (-1)
            gt_b_sum_2 += gt_b_2 ** 2
            gt_b_sum_avg_2 = gt_b_sum_2 / count
            gt_b_tot_2 = (gt_b_sum_avg_2 + epilson) ** (-0.5)
            lr = lrate * ((count) ** (-0.5))
            # update weight
            w -= lr * gt_w_tot_1 * gt_w_1
            w_lr -= lr * gt_w_tot_2 * gt_w_2

            # update bias
            b -= lr * gt_b_tot_1 * gt_b_1
            b_lr -= lr * gt_b_tot_2 * gt_b_2


    return w, b, w_lr, b_lr

def softmax(z1,z2):

    e1 = np.exp(z1)
    e2 = np.exp(z2)

    tot = np.sum(e1)
    y = np.zeros(e1.shape)

    # for i in range(e1.shape[0]):
    #     e1_pct = (e1[i])/(e1[i] + e2[i])
    #     e2_pct = (e2[i])/(e1[i] + e2[i])
    #     e1[i] = e1_pct
    #     e2[i] = e2_pct
    #     if(e1_pct > 0.5):
    #         y[i] = 1
    #     else:
    #         y[i] = 0

    for i in range(e1.shape[0]):
        e1_pct = (e1[i])/(tot)
        e1[i] = e1_pct
        if(e1_pct > 0.5):
            y[i] = 1
        else:
            y[i] = 0

    # print(e1)

    return y  

def test(x_test, w1, b1, w2, b2):
    
    
    z1 = np.dot(w1,x_test.transpose()) + b1
    z2 = np.dot(w2,x_test.transpose()) + b2
    # f1 = sigmoid(z1)
    z1 = sigmoid(z1)
    # print(z1)
    z2 = sigmoid(z2)
    
    y = np.zeros(z1.shape[0])

    # f = softmax(z1,z2)
    i = 0
    count = 0
    for num in z1:
        if num >= 0.5:
            y[i] = 1
            count += 1
        else:
            y[i] = 0      
        
        i += 1
    
    print("count is %d" %(count))

    return z1

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
    x_train, x_test = normalize(x_train, x_test)

    feats = [0, 2, 3, 5, 10, 33, 41, 53]
    # feats = [0, 2, 3, 4, 5, 24, 27, 29, 33, 41, 47, 53]
    # feats = [0, 2, 3, 4, 5, 10, 24, 25, 27, 29, 33, 41, 47, 53, 58]
    # feats = [0, 2, 3, 4, 5, 10, 24, 25, 27, 29, 33, 41, 47, 53, 58, 63]
    # feats = [0, 2, 3, 4, 5, 6, 7, 10, 11, 24, 25, 27, 29, 33, 41, 47, 48, 49, 50, 53, 58, 63, 82, 102]
    x_train = x_train[:,feats]
    x_test = x_test[:,feats]

    w1, b1, w2, b2= train(x_train,y_train)

    y_test = test(x_test, w1, b1, w2, b2)

    y = test(x_train, w1, b1, w2, b2)
    y = np.around(y) 

    result = testacc(y_train, y)

    print('Train acc = %f' % (result))

    with open('predict.csv', 'w', newline='') as csvf:
        # 建立 CSV 檔寫入器
        writer = csv.writer(csvf)
        writer.writerow(['id','label'])
        for i in range(int(y_test.shape[0])):
            writer.writerow( [i + 1, int(y_test[i])] )