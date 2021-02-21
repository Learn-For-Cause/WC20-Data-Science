import pandas as pd
import numpy as np
from geopy.distance import vincenty
import sys
import datetime
import math


start_time = datetime.datetime.now()


def new_user(x):
    if x == "NO":
        return 10
    if x == "YES":
        return 11
    else:
        return np.nan


def store_and_fwd_flag(x):
    if x == "N":
        return 0
    if x == "Y":
        return 1
    else:
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
test = pd.read_csv("test.csv", sep=",", header=0, usecols=range(2,17),
                   converters={'new_user': new_user,
                               "store_and_fwd_flag": store_and_fwd_flag,
                               "payment_type": payment_type})


# ------------------------------- Pre-Processing
print("Done reading. Pre-processing..")

#test = test["pickup_datetime"] != '' or test["pickup_datetime"] != 0
test["pickup_latitude"] = np.where(test["pickup_latitude"].isnull() | test["pickup_latitude"] == 0,
                                    np.nan, test["pickup_latitude"])
test["pickup_longitude"] = np.where(test["pickup_longitude"].isnull() | test["pickup_longitude"] == 0,
                                    np.nan, test["pickup_longitude"])
test["dropoff_latitude"] = np.where(test["dropoff_latitude"].isnull() | test["dropoff_latitude"] == 0,
                                    np.nan, test["dropoff_latitude"])
test["dropoff_longitude"] = np.where(test["dropoff_longitude"].isnull() | test["dropoff_longitude"] == 0,
                                    np.nan, test["dropoff_longitude"])
test["pickup_datetime"] = np.where(test["pickup_datetime"].isnull() | test["pickup_datetime"] == 0,
                                    np.nan, test["pickup_datetime"])
test["dropoff_datetime"] = np.where(test["dropoff_datetime"].isnull() | test["dropoff_datetime"] == 0,
                                    np.nan, test["dropoff_datetime"])

test.loc[
    test["pickup_latitude"].isnull() |
    test["pickup_longitude"].isnull() |
    test["dropoff_latitude"].isnull() |
    test["dropoff_longitude"].isnull(),
    ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']] = np.nan

test.loc[test["pickup_datetime"].isnull() |
          test["dropoff_datetime"].isnull(),
          ['pickup_datetime', 'dropoff_datetime']] = np.nan


# ------------------------------- Setting variables when fare_amount = 0
test["new_user"] = np.where(test["new_user"] == 11 | ((test["tip_amount"] < 0.0) | (test["surcharge"] < 0.0)
                              & (test["mta_tax"] < 0.0) | (test["tolls_amount"] < 0.0 )), 11, 10)
test.loc[test["new_user"] == 11, ['tip_amount','surcharge','mta_tax','tolls_amount']] = -1

print(test.shape)


# ------------------------------- Filling missing data
'''test["tolls_amount"] = test["tolls_amount"].fillna(test["tolls_amount"].median())
test["tip_amount"] = test["tip_amount"].fillna(test["tip_amount"].median())
test["mta_tax"] = test["mta_tax"].fillna(test["mta_tax"].median())
test["passenger_count"] = test["passenger_count"].fillna(test["passenger_count"].median())
test["rate_code"] = test["rate_code"].fillna(test["rate_code"].median())
test["store_and_fwd_flag"] = test["store_and_fwd_flag"].fillna(test["store_and_fwd_flag"].median())
test["surcharge"] = test["surcharge"].fillna(test["surcharge"].median())
'''

print(set(test["payment_type"]))

test["journey_time"] = pd.to_timedelta(pd.to_datetime(test["dropoff_datetime"], errors="coerce") - \
                        pd.to_datetime(test["pickup_datetime"], errors="coerce"))

test["journey_time"] = test.apply(lambda x: np.ceil(x["journey_time"].total_seconds()/60), axis=1)
test["journey_time"] = np.where(test["journey_time"] < 0, 0, test["journey_time"])
print(test.shape)
print("*"*30)



# ------------------------------- Extracting time details
#test["pickup_datetime"] = np.where() --- Condition for pickup<dropoff
test["travel_year"] = pd.to_datetime(test["pickup_datetime"], errors="coerce").dt.year
test["travel_month"] = pd.to_datetime(test["pickup_datetime"], errors="coerce").dt.month
#test["travel_ym"] = (pd.to_datetime(test["pickup_datetime"], errors="coerce").dt.year)*100 + \
#                     (pd.to_datetime(test["pickup_datetime"], errors="coerce").dt.month)
test["pickup_hour"] = pd.to_datetime(test["pickup_datetime"], errors="coerce").dt.hour
test["dropoff_hour"] = pd.to_datetime(test["dropoff_datetime"], errors="coerce").dt.hour

# ------------------------------- Extracting distance
test["src"] = list(zip(test["pickup_latitude"],test["pickup_longitude"]))
test["dest"] = list(zip(test["dropoff_latitude"],test["dropoff_longitude"]))
test["distance"] = test.apply(lambda x: np.nan if (np.isnan(x["src"][0])) else (round(vincenty(x["src"],x["dest"]).miles, ndigits=2)), axis=1)
test["distance"] = test["distance"].fillna(test["distance"].median())
print(test.shape)
#test.dropna()
print(test.shape)

# ------------------------------- Dropping unnecessary columns
test.drop(["pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude",
            "pickup_datetime", "dropoff_datetime", "src", "dest"], axis=1, inplace=True)
print(test)
#test = test.dropna()
print(test.shape)

test.to_csv("Proc_test2.csv", index=False)

print("Time taken: " + str(datetime.datetime.now() - start_time))