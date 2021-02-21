import sys
import datetime
import pandas as pd
import numpy as np
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

'''
Testing model on validation set
'''

start_time = datetime.datetime.now()

# ------------------------------------------------- Reading files

print("Reading file..")
train = pd.read_csv("C:\\Users\\satvi\\Documents\\Projects\\Hackerearth - ML 2\\Data Files\\final_proc_train_try1.csv",
                    sep=",", header=0)
pred = pd.read_csv("C:\\Users\\satvi\\Documents\\Projects\\Hackerearth - ML 2\\Data Files\\test.csv",
                   sep=",", header=0)
train2 = pd.read_csv("C:\\Users\\satvi\\Documents\\Projects\\Hackerearth - ML 2\\Data Files\\train.csv",
                   sep=",", header=0)
w1 = pd.read_csv("C:\\Users\\satvi\\Documents\\Projects\\Hackerearth - ML 2\\Data Files\\test_words_in_train.csv",
                   sep=",", header=0)
w2 = pd.read_csv("C:\\Users\\satvi\\Documents\\Projects\\Hackerearth - ML 2\\Data Files\\test_words_in_test.csv",
                   sep=",", header=0)
w3 = pd.read_csv("C:\\Users\\satvi\\Documents\\Projects\\Hackerearth - ML 2\\Data Files\\train_words_in_train.csv",
                   sep=",", header=0)
w4 = pd.read_csv("C:\\Users\\satvi\\Documents\\Projects\\Hackerearth - ML 2\\Data Files\\train_words_in_test.csv",
                   sep=",", header=0)
w5 = pd.read_csv("C:\\Users\\satvi\\Documents\\Projects\\Hackerearth - ML 2\\Data Files\\test_keywords_in_train.csv",
                   sep=",", header=0)
w6 = pd.read_csv("C:\\Users\\satvi\\Documents\\Projects\\Hackerearth - ML 2\\Data Files\\test_keywords_in_test.csv",
                   sep=",", header=0)
w7 = pd.read_csv("C:\\Users\\satvi\\Documents\\Projects\\Hackerearth - ML 2\\Data Files\\train_keywords_in_train.csv",
                   sep=",", header=0)
w8 = pd.read_csv("C:\\Users\\satvi\\Documents\\Projects\\Hackerearth - ML 2\\Data Files\\train_keywords_in_test.csv",
                   sep=",", header=0)
w9 = pd.read_csv("C:\\Users\\satvi\\Documents\\Projects\\Hackerearth - ML 2\\Data Files\\count_key_train.csv",
                   sep=",", header=0)
w10 = pd.read_csv("C:\\Users\\satvi\\Documents\\Projects\\Hackerearth - ML 2\\Data Files\\count_key_test.csv",
                   sep=",", header=0, nrows=63465)
w11 = pd.read_csv("C:\\Users\\satvi\\Documents\\Projects\\Hackerearth - ML 2\\Data Files\\W2V_train.csv",
                   sep=",", header=0)
w12 = pd.read_csv("C:\\Users\\satvi\\Documents\\Projects\\Hackerearth - ML 2\\Data Files\\W2V_test.csv",
                   sep=",", header=0, nrows=63465)

print(train)
train = pd.concat([train,w1], axis=1)
train = pd.concat([train,w3], axis=1)
train = pd.concat([train,w5], axis=1)
train = pd.concat([train,w7], axis=1)
train = pd.concat([train,w9], axis=1)
# train = pd.concat([train,w11], axis=1)
print(train.shape)
train = train.loc[:,~train.columns.duplicated()]
print(train.shape)
train_mod = train[train["final_status"] == 1]
train = train.append(train_mod, ignore_index=True)
Y = train["final_status"].as_matrix()
train.drop(["created_at_hour","fund_duration_sec","fund_duration_min","project_id","disable_communication","final_status"], axis=1, inplace=True)
X = train.as_matrix()


# ------------------------------------------------- Performing Simple Train Test Split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=69)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
print("Scaling..")
scaler = StandardScaler()
X_train_transformed = scaler.fit_transform(X_train)
X_test_transformed = scaler.transform(X_test)

'''from sklearn.decomposition import PCA
pca = PCA(n_components=None)
X_train_transformed = pca.fit_transform(X_train_transformed)
X_test_transformed = pca.transform(X_test_transformed)
print(pca.components_)
# --PCA 0.69075 476
'''
print("Running XGB Classifier...")
classifier2 = XGBClassifier(n_estimators=2000, nthread=2, seed=69)
model2 = classifier2.fit(X_train_transformed,y_train)
print(model2.score(X_test_transformed, y_test))
print(model2.feature_importances_)
print(train.columns.tolist())
# '''

# -------------------- Performing K-Fold CV
'''
cv = KFold(n_splits=3,random_state=69, shuffle=True)
results = []
classifier2 = XGBClassifier(n_estimators=2000, nthread=2, seed=69)
X = X.as_matrix()
# print(X)
# X = StandardScaler().fit(X).transform(X)
# print(X)
Y = Y.as_matrix()
for train_index, test_index in cv.split(X=X, y=Y):
    print(train_index)
    print(test_index)
    pred = classifier2.fit(X[train_index], Y[train_index]).score(X[test_index], Y[test_index])
    results.append(pred)
print(np.mean(results))
'''
print("Time taken: " + str(datetime.datetime.now() - start_time))


