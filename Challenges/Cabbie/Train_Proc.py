import pandas as pd
import numpy as np
from geopy.distance import vincenty
import sys
import datetime

from pandas.io.sas.sas_constants import index

start_time = datetime.datetime.now()


def new_user(x):
    if x == "NO":
        return 10
    if x == "YES":
        return 11
    return np.nan


def store_and_fwd_flag(x):
    if x == "N":
        return 0
    if x == "Y":
        return 1
    return np.nan


def payment_type(x):
    if x == "CRD":
        return 0
    elif x == "CSH":
        return 1
    elif x == "UNK":
        return 2
    elif x == "NOC":
        return 3
    elif x == "DIS":
        return 4
    return 5

'''
def passenger_count(x):
    if int(x)<=4:
        return 0
    elif int(x)>4 and int(x)<=6:
        return 1
    elif int(x)>6:
        return 2
    return np.nan
'''
print("Reading file..")
train = pd.read_csv("train.csv", sep=",", header=0, usecols=range(2,18),
                    converters={'new_user':new_user ,
                                "store_and_fwd_flag":store_and_fwd_flag,
                                "payment_type":payment_type})


# ------------------------------- Pre-Processing
print("Done reading. Pre-processing..")

#train = train["pickup_datetime"] != '' or train["pickup_datetime"] != 0
train["pickup_latitude"] = train["pickup_latitude"].replace(0,np.nan)
train["pickup_longitude"] = train["pickup_longitude"].replace(0,np.nan)
train["dropoff_latitude"] = train["dropoff_latitude"].replace(0,np.nan)
train["dropoff_longitude"] = train["dropoff_longitude"].replace(0,np.nan)
train["pickup_datetime"] = train["pickup_datetime"].replace(0,np.nan)
train["dropoff_datetime"] = train["dropoff_datetime"].replace(0,np.nan)
print(train.shape)

train.loc[
    train["pickup_latitude"].isnull() |
    train["pickup_longitude"].isnull() |
    train["dropoff_latitude"].isnull() |
    train["dropoff_longitude"].isnull(),
    ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']] = np.nan

# ------------------------------- Setting variables when fare_amount = 0
train["new_user"] = np.where(train["fare_amount"] == 0, 11, 10)
train.loc[train["fare_amount"] == 0,['tip_amount','surcharge','mta_tax','tolls_amount']] = -1

# ------------------------------- Filling missing data
'''train["tolls_amount"] = train["tolls_amount"].fillna(train["tolls_amount"].median())
train["tip_amount"] = train["tip_amount"].fillna(train["tip_amount"].median())
train["mta_tax"] = train["mta_tax"].fillna(train["mta_tax"].median())
train["passenger_count"] = train["passenger_count"].fillna(train["passenger_count"].median())
train["rate_code"] = train["rate_code"].fillna(train["rate_code"].median())
train["store_and_fwd_flag"] = train["store_and_fwd_flag"].fillna(train["store_and_fwd_flag"].median())
train["surcharge"] = train["surcharge"].fillna(train["surcharge"].median())
'''
print(set(train["payment_type"]))
print("*"*30)
#train = train.dropna()
print(train.shape)

train["journey_time"] = pd.to_timedelta(pd.to_datetime(train["dropoff_datetime"], errors="coerce") - \
                        pd.to_datetime(train["pickup_datetime"], errors="coerce"))
train["journey_time"] = train.apply(lambda x: np.ceil(x["journey_time"].total_seconds()/60), axis=1)
train["journey_time"] = np.where(train["journey_time"] < 0, np.nan, train["journey_time"])

# ------------------------------- Extracting time details
#train["pickup_datetime"] = np.where() --- Condition for pickup<dropoff
train["travel_year"] = pd.to_datetime(train["pickup_datetime"], errors="coerce").dt.year
train["travel_month"] = pd.to_datetime(train["pickup_datetime"], errors="coerce").dt.month
#train["travel_ym"] = (pd.to_datetime(train["pickup_datetime"], errors="coerce").dt.year)*100 + \
#                     (pd.to_datetime(train["pickup_datetime"], errors="coerce").dt.month)
train["pickup_hour"] = pd.to_datetime(train["pickup_datetime"], errors="coerce").dt.hour
train["dropoff_hour"] = pd.to_datetime(train["dropoff_datetime"], errors="coerce").dt.hour

# ------------------------------- Extracting distance
train["src"] = list(zip(train["pickup_latitude"],train["pickup_longitude"]))
train["dest"] = list(zip(train["dropoff_latitude"],train["dropoff_longitude"]))
train["distance"] = train.apply(lambda x: np.nan if (np.isnan(x["src"][0])) else (round(vincenty(x["src"],x["dest"]).miles, ndigits=2)), axis=1)
print(train.shape)

# ------------------------------- Dropping unnecessary columns
train.drop(["pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude",
            "pickup_datetime", "dropoff_datetime", "src", "dest"], axis=1, inplace=True)

#train = train.dropna()
print(train.shape)


cols = train.columns.tolist()
cols.insert(len(cols), cols.pop(cols.index('fare_amount')))
print(cols)
train = train.reindex(columns= cols)

train.to_csv("Proc_train.csv", index=False)

print("Time taken: " + str(datetime.datetime.now() - start_time))