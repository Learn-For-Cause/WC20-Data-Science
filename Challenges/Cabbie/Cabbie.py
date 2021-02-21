import pandas as pd
import numpy as np
from geopy.distance import vincenty
import sys
from sklearn.ensemble import RandomForestRegressor
import datetime
from sklearn.model_selection import train_test_split
import xgboost

from pandas.io.sas.sas_constants import index

start_time = datetime.datetime.now()

print("Reading file..")
train = pd.read_csv("Proc_train.csv", sep=",", header=0)
test = pd.read_csv("Proc_test2.csv", sep=",", header=0)
pred = pd.read_csv("test.csv", sep=",", header=0, usecols=range(0,1))

print(train.shape)
print(test.shape)
Y = train["fare_amount"]
X = train.drop("fare_amount" , axis=1)
del train
print(X.columns.tolist())
'''X, val, Y, Y_val = train_test_split(X,Y, test_size=0.2, random_state=69)
print(X.shape)
print(val.shape)
print(Y.shape)
print(Y_val.shape)
'''
# ------------------------------ RandomForestRegressor
'''regressor = RandomForestRegressor(n_estimators=800, criterion="friedman_mse", n_jobs=3, random_state =50,
                                     max_features="auto", min_samples_leaf=5)
regressor.fit(X,Y)
print("Done training. Beginning prediction.")
prediction = regressor.predict(test)
print(prediction)
'''
# ------------------------------ XGBoostRegressor
xgregressor1 = xgboost.XGBRegressor(n_estimators=100, nthread=3, seed=50)
xgregressor1.fit(X,Y)
'''xgregressor2 = xgboost.XGBRegressor(n_estimators=1500, nthread=3, reg_alpha=0.005, colsample_bytree=0.6,
                                   learning_rate=0.025, subsample=0.8, seed=50)
xgregressor2.fit(X,Y)
#print(xgregressor.booster().get_score(importance_type='gain'))
print("Done training. Beginning prediction.")
prediction1 = xgregressor1.predict(val)
prediction2 = xgregressor2.predict(val)
print(Y_val - prediction1)
val["actual"] = Y_val
val["predicted1"] = prediction1
val["diff1"] = Y_val - prediction1
val["predicted2"] = prediction2
val["diff2"] = Y_val - prediction2
val["index"] = val.index.values
val.loc[val["new_user"] == 1, ["predicted1","predicted2"]] = 0
val["diff1"] = val["actual"] - val["predicted1"]
val["diff2"] = val["actual"] - val["predicted2"]
val.sort_index(inplace=True)
cols = val.columns.tolist()
np.savetxt("validation2.csv", val, delimiter=',', fmt="%f", header=str(cols), comments='')
print("Time taken: " + str(datetime.datetime.now() - start_time))
sys.exit()
'''

prediction = xgregressor1.predict(test)
pred["fare_amount"] = prediction
pred.loc[test["new_user"] == 1, ["fare_amount"]] = 0
np.savetxt("prediction.csv", pred, delimiter=',', fmt=["%s","%f"], header="TID, fare_amount", comments='')
print("Time taken: " + str(datetime.datetime.now() - start_time))