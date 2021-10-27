import pandas as pd

y = pd.read_csv('predict_1.csv')
y = y.values
y = y.reshape(-1)
count = 0
for num in y :
    if(num == 1) :
        count += 1
print(count)