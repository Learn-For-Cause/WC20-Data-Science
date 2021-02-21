import numpy as np
import pandas as pd
import sys
from scipy.stats import mode
import random

file1 = pd.read_csv(".\\Predictions\\Predict2.csv", header=0)
file2 = pd.read_csv(".\\Predictions\\Predict14.csv", header=0)
file3 = pd.read_csv(".\\Predictions\\Predict13.csv", header=0)
# file4 = pd.read_csv(".\\Predictions\\Predict13.csv", header=0)
# file5 = pd.read_csv(".\\Predictions\\Predict8.csv", header=0)
file1 = file1.sort_values(['PID']).reset_index(drop=True)
file2 = file2.sort_values(['PID']).reset_index(drop=True)
file3 = file3.sort_values(['PID']).reset_index(drop=True)
# file4 = file4.sort_values(['PID']).reset_index(drop=True)
# file5 = file5.sort_values(['PID']).reset_index(drop=True)
# print(file1)
cols = file1.columns.tolist()
print(cols)
matrix = list()

for i in range(0,3000):
    df = pd.DataFrame(columns=cols)
    df = df.append(file1.iloc[i], ignore_index=True)
    df = df.append(file2.iloc[i], ignore_index=True)
    df = df.append(file3.loc[i], ignore_index=True)
    # df = df.append(file4.loc[i], ignore_index=True)
    # df = df.append(file5.loc[i], ignore_index=True)
    rlist = [df['Event1'].iloc[0], df['Event1'].iloc[1], df['Event1'].iloc[2],
             df['Event2'].iloc[0], df['Event2'].iloc[1], df['Event2'].iloc[2]]
    df['PID'] = df['PID'].astype(int)
    new_row = list(df.mode().values[0])
    for j in range(1,11):
        if pd.isnull(new_row[j]):
            new_row[j] = random.choice(rlist)
    print(new_row)
    matrix.append(new_row)
    del df


df_final = pd.DataFrame(matrix, columns=cols)
df_final['PID'] = df_final['PID'].astype(int)
print(df_final)

df_final.to_csv(".\\Predictions\\Predict16.csv", index=False)