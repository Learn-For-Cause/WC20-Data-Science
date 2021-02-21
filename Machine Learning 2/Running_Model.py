import sys
import datetime
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from xgboost.sklearn import XGBClassifier
from xgboost.sklearn import XGBModel

'''
Running model on test set to get predictions
'''

start_time = datetime.datetime.now()

# ------------------------------------------------- Reading files

print("Reading files..")
train = pd.read_csv("C:\\Users\\satvi\\Documents\\Projects\\Hackerearth - ML 2\\Data Files\\final_proc_train_try1.csv",
                    sep=",", header=0)
test = pd.read_csv("C:\\Users\\satvi\\Documents\\Projects\\Hackerearth - ML 2\\Data Files\\final_proc_test_try1.csv",
                   sep=",", header=0)
pred = pd.read_csv("C:\\Users\\satvi\\Documents\\Projects\\Hackerearth - ML 2\\Data Files\\test.csv",
                   sep=",", header=0, usecols=range(0,1))
train2 = pd.read_csv("C:\\Users\\satvi\\Documents\\Projects\\Hackerearth - ML 2\\Data Files\\train.csv",
                   sep=",", header=0)
test2 = pd.read_csv("C:\\Users\\satvi\\Documents\\Projects\\Hackerearth - ML 2\\Data Files\\test.csv",
                   sep=",", header=0)
w1 = pd.read_csv("C:\\Users\\satvi\\Documents\\Projects\\Hackerearth - ML 2\\Data Files\\test_words_in_train.csv",
                   sep=",", header=0)
w2 = pd.read_csv("C:\\Users\\satvi\\Documents\\Projects\\Hackerearth - ML 2\\Data Files\\test_words_in_test.csv",
                   sep=",", header=0)
w3 = pd.read_csv("C:\\Users\\satvi\\Documents\\Projects\\Hackerearth - ML 2\\Data Files\\train_words_in_train.csv",
                   sep=",", header=0)
w4 = pd.read_csv("C:\\Users\\satvi\\Documents\\Projects\\Hackerearth - ML 2\\Data Files\\train_words_in_test.csv",
                   sep=",", header=0, nrows=63465)
w5 = pd.read_csv("C:\\Users\\satvi\\Documents\\Projects\\Hackerearth - ML 2\\Data Files\\test_keywords_in_train.csv",
                   sep=",", header=0)
w6 = pd.read_csv("C:\\Users\\satvi\\Documents\\Projects\\Hackerearth - ML 2\\Data Files\\test_keywords_in_test.csv",
                   sep=",", header=0)
w7 = pd.read_csv("C:\\Users\\satvi\\Documents\\Projects\\Hackerearth - ML 2\\Data Files\\train_keywords_in_train.csv",
                   sep=",", header=0)
w8 = pd.read_csv("C:\\Users\\satvi\\Documents\\Projects\\Hackerearth - ML 2\\Data Files\\train_keywords_in_test.csv",
                   sep=",", header=0, nrows=63465)
w9 = pd.read_csv("C:\\Users\\satvi\\Documents\\Projects\\Hackerearth - ML 2\\Data Files\\count_key_train.csv",
                   sep=",", header=0)
w10 = pd.read_csv("C:\\Users\\satvi\\Documents\\Projects\\Hackerearth - ML 2\\Data Files\\count_key_test.csv",
                   sep=",", header=0, nrows=63465)
w11 = pd.read_csv("C:\\Users\\satvi\\Documents\\Projects\\Hackerearth - ML 2\\Data Files\\W2V_train.csv",
                   sep=",", header=0)
w12 = pd.read_csv("C:\\Users\\satvi\\Documents\\Projects\\Hackerearth - ML 2\\Data Files\\W2V_test.csv",
                   sep=",", header=0, nrows=63465)
w13 = pd.read_csv("C:\\Users\\satvi\\Documents\\Projects\\Hackerearth - ML 2\\Data Files\\count_unique_train.csv",
                   sep=",", header=0)
w14 = pd.read_csv("C:\\Users\\satvi\\Documents\\Projects\\Hackerearth - ML 2\\Data Files\\count_unique_test.csv",
                   sep=",", header=0, nrows=63465)

train = pd.concat([train,w1], axis=1)
train = pd.concat([train,w3], axis=1)
train = pd.concat([train,w5], axis=1)
train = pd.concat([train,w7], axis=1)
train = pd.concat([train,w9], axis=1)
# train = pd.concat([train,w11], axis=1)
# train = pd.concat([train,w13], axis=1)
print(train.shape)
train = train.loc[:,~train.columns.duplicated()]
print(train.shape)

test = pd.concat([test,w2], axis=1)
test = pd.concat([test,w4], axis=1)
test = pd.concat([test,w6], axis=1)
test = pd.concat([test,w8], axis=1)
test = pd.concat([test,w10], axis=1)
# test = pd.concat([test,w12], axis=1)
# test = pd.concat([test,w14], axis=1)
print(test.shape)
test = test.loc[:,~test.columns.duplicated()]
print(test.shape)
print(test)
train_mod = train[train["final_status"] == 1]
train = train.append(train_mod, ignore_index=True)
print(train[train["final_status"] == 1].shape)
print(train[train["final_status"] == 0].shape)
print(train)

#test['created_at'] = test2['created_at']
#test['launched_at'] = test2['launched_at']
Y = train["final_status"].as_matrix()
train.drop(["fund_duration_sec","fund_duration_min","project_id","disable_communication","final_status"], axis=1, inplace=True)
#test.drop(["name","desc","keywords"], axis=1, inplace=True)
X = train.as_matrix()
test.drop(["fund_duration_sec","fund_duration_min","project_id","disable_communication"],  axis=1, inplace=True)
del train
test = test.as_matrix()
print(X.shape)
print(test.shape)

# ------------------------------------------------- Applying PCA
"""
print("Applying PCA...")
from sklearn.decomposition import PCA
pca = PCA(n_components=333)
X = pca.fit_transform(X)
test = pca.transform(test)
print(pca.components_)
"""


# ------------------------------------------------- Building XGB Model

classifier2 = XGBClassifier(n_estimators=2500, silent=False, nthread=-1, seed=69, max_depth=4)
model2 = classifier2.fit(X,Y)
prediction2 = model2.predict(test)

print(prediction2)
pred['final_status'] = prediction2

np.savetxt("prediction.csv",
           pred, delimiter=',', fmt=["%s","%i"], header="project_id, final_status", comments='')
print("Time taken: " + str(datetime.datetime.now() - start_time))
print(model2.feature_importances_)

# ------------------------------------------------- Performance Evaluation
'''
XGBClassifier(n_estimators=2000, nthread=2, seed=69) :  Validation: 0.68535 / Test- try1: 0.68685
XGBClassifier(n_estimators=355, nthread=2, seed=69) :  Validation: 0.69 / Test- try1: 0.68382
XGBClassifier(n_estimators=2000, nthread=2, seed=69) :  Validation: 0.6888 / Test- try2: 0.68527
XGBClassifier(n_estimators=355, nthread=2, seed=69) :  Validation: 0.69111 / Test- try2: 0.68316
XGBClassifier(n_estimators=2000, nthread=2, seed=69) :  Validation: 0.69065 / Test- try1: 0.68826 --month,day
XGBClassifier(n_estimators=2000, nthread=2, seed=69) :  Validation: 0.69544 / Test- try1: 0.67752 --year
XGBClassifier(n_estimators=2000, nthread=2, seed=69) :  Validation: 0.65132 / Test- try1: 0.70257 --month,day, train(1)x2
XGBClassifier(n_estimators=2000, nthread=2, seed=69) :  Validation: 0.65954 / Test- try1: 0.71177 --month,day, train(1)x2, dc
XGBClassifier(n_estimators=2000, nthread=2, seed=69) :  Validation: 0.6615 / Test- try1: 0.71083 --month,day, train(1)x2, dc, lc
XGBClassifier(n_estimators=2000, nthread=2, seed=69) :  Validation: 0.692 / Test- try1: 0.73097 --month,day, train(1)x2, dc, tw 200
XGBClassifier(n_estimators=2000, nthread=2, seed=69) :  Validation: 0.6939 / Test- try1: 0.73654 --month,day, train(1)x2, dc, tw 350
XGBClassifier(n_estimators=2000, nthread=2, seed=69) :  Validation: 0.69932 / Test- try1: 0.73522 --month,day, train(1)x2, dc, tw 450
XGBClassifier(n_estimators=2000, nthread=2, seed=69) :  Validation: 0.70432 / Test- try1: 0.73922 --month,day, train(1)x2, dc, tw 350 key 150
XGBClassifier(n_estimators=2000, nthread=2, seed=69) :  Validation: 0.70432 / Test- try1: 0.73932 --month,day, train(1)x2, dc, tw 350 key 100
XGBClassifier(n_estimators=2000, nthread=2, seed=69) :  Validation: ? / Test- try1: 0.73692 --month,day, train(1)x2, dc, tw 400
XGBClassifier(n_estimators=2000, nthread=2, seed=69) :  Validation: 0.70215 / Test- try1: 0.74092 --month,day, train(1)x2, dc, tw 400 key 100
XGBClassifier(n_estimators=2000, nthread=2, seed=69) :  Validation: ? / Test- try1: 0.74102 --month,day, train(1)x2, dc, tw 400 key 100 cnt 300
XGBClassifier(n_estimators=2000, nthread=2, seed=69) :  Validation: ? / Test- try1: 0.74051 --month,day, train(1)x2, dc, tw 400 key 100 cnt 200
XGBClassifier(n_estimators=2000, nthread=2, seed=69) :  Validation: ? / Test- try1: 0.74215 --month,day, train(1)x2, dc, tw 400 key 100 cnt 400
XGBClassifier(n_estimators=2000, nthread=2, seed=69) :  Validation: 0.70685 / Test- try1: 0.74291 --month,day, train(1)x2, dc, tw 400 key 100 cnt 500
XGBClassifier(n_estimators=2000, nthread=2, seed=69) :  Validation: ? / Test- try1: 0.74168 --month,day, train(1)x2, dc, tw 400 key 100 cnt 600
XGBClassifier(n_estimators=2000, nthread=2, seed=69) :  Validation: ? / Test- try1: 0.74140 --month,day, train(1)x2, dc, tw 400 key 100 cnt 475
XGBClassifier(n_estimators=2500, nthread=2, seed=69) :  Validation: ? / Test- try1: 0.74313 --month,day, train(1)x2, dc, tw 400 key 100 cnt 500
XGBClassifier(n_estimators=2000, nthread=2, seed=69, max_depth=5, colsample_bytree=0.7, subsample=0.7, min_child_weight=5) :  Validation: 0.73203 / Test- try1: 0.73966 --month,day, train(1)x2, dc, tw 400 key 100 cnt 500
XGBClassifier(n_estimators=2000, nthread=2, seed=69, colsample_bytree=0.7, subsample=0.7, min_child_weight=5) :  Validation: 0.70715 / Test- try1: 0.74231 --month,day, train(1)x2, dc, tw 400 key 100 cnt 500
'''