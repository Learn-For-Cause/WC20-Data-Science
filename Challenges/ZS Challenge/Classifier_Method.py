import numpy as np
import pandas as pd
import datetime
import sys
import os
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

start_time = datetime.datetime.now();
print("Started at: "+ str(start_time));


def convert_date(x):
    return int(x[0:4])

print(os.getcwd())
train = pd.read_csv(".\\Datasets\\train.csv", converters= {'Date':convert_date})
print(train.shape)
train = train[~train['Event'].str.contains('J') & ~train['Event'].str.contains('G') & ~train['Event'].str.contains('A')
              & ~train['Event'].str.contains('S') & ~train['Event'].str.contains('Q')]
print(train.shape)
#train.drop('Date', axis=1, inplace=True)
prediction_matrix = list()

def predictions(x):
    global train
    print(x)
    train_copy = train[train['PID'] == int(x)]
    prediction_list = list()
    for i in range(1,11):
        Y = train_copy['Event'].as_matrix()
        X = train_copy.copy()
        X.drop('Event', axis=1, inplace=True)
        test = pd.DataFrame(columns=['PID','Date'], data={'PID':int(x),'Date': 2014}, index=[0])
        classifier = RandomForestClassifier(n_estimators=13, random_state=145)
        #classifier = XGBClassifier(n_estimators=21, seed=145)
        model = classifier.fit(X,Y)
        pred = model.predict(test)
        prediction_list.extend(pred)
        print(pred)
        train_copy = train_copy[train_copy['Event'] != pred[0]]
    prediction_matrix.append(prediction_list)
    return x


test = pd.read_csv(".\\Datasets\\test.csv", converters= {'PID':predictions})

print(prediction_matrix)
columns = ["Event"+str(i) for i in range(1,11)]
df1 = pd.DataFrame(prediction_matrix, columns=columns)
df1['PID'] = test['PID']
columns.insert(0,'PID')
df1 = df1.reindex(columns=columns)
df1.to_csv(".\\Predictions\\Predict14.csv", index= False)
print("Total time taken: "+ str(datetime.datetime.now() - start_time))
