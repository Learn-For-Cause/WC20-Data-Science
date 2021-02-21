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
train = pd.read_csv(".\\Datasets\\train.csv")
train = train[~train['Event'].str.contains('J') & ~train['Event'].str.contains('G') & ~train['Event'].str.contains('A')
              & ~train['Event'].str.contains('S') & ~train['Event'].str.contains('Q')]
prediction_matrix = list()
train['index'] = train.index
train = train.sort_values(['Date', 'index'])
events = list(set(train['Event']))
pid = list(set(train['PID']))
df = pd.DataFrame()
df['Event'] = events
df.set_index('Event', inplace=True)
for e in events:
    df[e] = 0
'''

# train_copy = train[train['PID'] == int(1028890)].copy()
# print(train_copy)
# train_copy['index'] = train_copy.index
# train_copy = train_copy.sort_values(['Date', 'index'])
# print(train_copy)
# y = train_copy[-1:]['Event']
print(events)
print(df.loc['9201']['9201'])
for id in pid:
    train_copy = train[train['PID'] == int(id)].copy()
    print(train_copy)
    x = train_copy['Event'].iloc[0]
    for start_event in train_copy[1:]['Event']:
        df.loc[x][start_event] += 1
        x = start_event
print(df)
print(df.loc['8501'])
print(df.loc['8501'].idxmax(axis=1))
'''
# sys.exit(1)

def predictions(x):
    global train
    print(x)
    train_copy = train[train['PID'] == int(x)].copy()
    # train_copy['index'] = train_copy.index
    # train_copy = train_copy.sort_values(['Date', 'index'])
    # events = list(set(train_copy['Event']))
    # df = pd.DataFrame()
    # df['Event'] = events
    # df.set_index('Event', inplace=True)
    # for e in events:
    #     df[e] = 0
    event1 = train_copy['Event'].iloc[0]
    y = train_copy['Event'].iloc[-1]
    for row in train_copy[1:].iterrows():
        if int(str(row[1][1])[0:4]) == 2011:
            df.loc[event1][row[1][2]] += 1
        elif int(str(row[1][1])[0:4]) == 2012:
            df.loc[event1][row[1][2]] += 2
        elif int(str(row[1][1])[0:4]) == 2013:
            df.loc[event1][row[1][2]] += 3
        event1 = row[1][2]
    prediction_list = list()
    for i in range(1,11):
        y = df.loc[y].idxmax(axis=1)
        prediction_list.append(y)
        # print(pred)
        # df[y][pred] += 1
        # y = pred
    prediction_matrix.append(prediction_list)
    print(prediction_list)
    return x


test = pd.read_csv(".\\Datasets\\test.csv", converters= {'PID':predictions})

print(prediction_matrix)
columns = ["Event"+str(i) for i in range(1,11)]
df1 = pd.DataFrame(prediction_matrix, columns=columns)
df1['PID'] = test['PID']
columns.insert(0,'PID')
df1 = df1.reindex(columns=columns)
df1.to_csv(".\\Predictions\\Predict7.csv", index= False)
print("Total time taken: "+ str(datetime.datetime.now() - start_time))
