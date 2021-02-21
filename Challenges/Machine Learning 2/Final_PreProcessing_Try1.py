from _operator import concat

import numpy as np
import pandas as pd
import sys
from sklearn.preprocessing import LabelEncoder

# -------------- Final Pre Processing of train & test data set - TRY 1
# - Replacing attributes name, desc and keywords with their counts
# - Binarization of attributes
# - Dropping all time stamps
# - Dropping currency, backers_count & backer_ratio
# --------------


def calc_words_name_desc(x):
    return len(str(x).split(' '))


def calc_words_keywords(x):
    return len(str(x).split('-'))

train = pd.read_csv("C:\\Users\\satvi\\Documents\\Projects\\Hackerearth - ML 2\\Data Files\\initial_proc_train.csv",
                    header=0, encoding='latin-1',
                    converters={'name': calc_words_name_desc,
                                'desc': calc_words_name_desc,
                                'keywords': calc_words_keywords})

test = pd.read_csv("C:\\Users\\satvi\\Documents\\Projects\\Hackerearth - ML 2\\Data Files\\initial_proc_test.csv",
                    header=0, encoding='latin-1',
                    converters={'name': calc_words_name_desc,
                                'desc': calc_words_name_desc,
                                'keywords': calc_words_keywords})


# train['launched_year'] = pd.to_datetime(train['launched_at']).dt.year
# train['deadline_year'] = pd.to_datetime(train['deadline']).dt.year
train['launched_month'] = pd.to_datetime(train['launched_at']).dt.month
train['deadline_month'] = pd.to_datetime(train['deadline']).dt.month
train['launched_day'] = pd.to_datetime(train['launched_at']).dt.day
train['deadline_day'] = pd.to_datetime(train['deadline']).dt.day
train['created_at_day'] = pd.to_datetime(train['created_at']).dt.day
train['created_at_hour'] = pd.to_datetime(train['created_at']).dt.hour
train['state_changed_at_day'] = pd.to_datetime(train['state_changed_at']).dt.day
# test['launched_year'] = pd.to_datetime(test['launched_at']).dt.year
# test['deadline_year'] = pd.to_datetime(test['deadline']).dt.year
test['launched_month'] = pd.to_datetime(test['launched_at']).dt.month
test['deadline_month'] = pd.to_datetime(test['deadline']).dt.month
test['launched_day'] = pd.to_datetime(test['launched_at']).dt.day
test['deadline_day'] = pd.to_datetime(test['deadline']).dt.day
test['created_at_day'] = pd.to_datetime(test['created_at']).dt.day
test['created_at_hour'] = pd.to_datetime(test['created_at']).dt.hour
test['state_changed_at_day'] = pd.to_datetime(test['state_changed_at']).dt.day


train['fund_duration_hrs_dc'] = pd.to_timedelta(pd.to_datetime(train['deadline']) - pd.to_datetime(train['created_at']))
train['fund_duration_hrs_dc'] = train.apply(lambda x: np.ceil(x['fund_duration_hrs_dc'].total_seconds()/3600), axis=1)

test['fund_duration_hrs_dc'] = pd.to_timedelta(pd.to_datetime(test['deadline']) - pd.to_datetime(test['created_at']))
test['fund_duration_hrs_dc'] = test.apply(lambda x: np.ceil(x['fund_duration_hrs_dc'].total_seconds()/3600), axis=1)


train.drop(['currency', 'deadline', 'state_changed_at', 'created_at', 'launched_at', 'backers_count'],
           axis=1, inplace=True)

test.drop(['currency', 'deadline', 'state_changed_at', 'created_at', 'launched_at'],
           axis=1, inplace=True)

binarize_features = ['disable_communication', 'country', 'canceled']
for x in binarize_features:
    le = LabelEncoder()
    le.fit(list(train[x].values) + list(test[x].values))
    train[x] = le.transform(list(train[x]))
    test[x] = le.transform(list(test[x]))

print(train)
print(test)

cols = train.columns.tolist()
cols.insert(len(cols), cols.pop(cols.index('final_status')))
print(cols)
train = train.reindex(columns= cols)

train.to_csv("C:\\Users\\satvi\\Documents\\Projects\\Hackerearth - ML 2\\Data Files\\final_proc_train_try1.csv",
             index=False)
test.to_csv("C:\\Users\\satvi\\Documents\\Projects\\Hackerearth - ML 2\\Data Files\\final_proc_test_try1.csv",
            index=False)