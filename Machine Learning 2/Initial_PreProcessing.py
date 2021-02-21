import numpy as np
import pandas as pd
import time
import datetime
import sys
from forex_python.converter import CurrencyRates
from currency_converter import CurrencyConverter


# -------------- Initial Pre Processing of train & test data set
# - Converting unix time
# - Regularizing Goal to 'USD'
# - Calculating duration of fund
# - Estimating if the fund was canceled/terminated before deadline
# - Calculating backer ratio to check for any patterns
# --------------


def unix_time_converter(x):
    return time.ctime(int(x))

train = pd.read_csv("C:\\Users\\satvi\\Documents\\Projects\\Hackerearth - ML 2\\Data Files\\train.csv",
                    header=0,
                    converters={'deadline':unix_time_converter,
                                'state_changed_at':unix_time_converter,
                                'created_at':unix_time_converter,
                                'launched_at':unix_time_converter})

test = pd.read_csv("C:\\Users\\satvi\\Documents\\Projects\\Hackerearth - ML 2\\Data Files\\test.csv",
                    header=0,
                    converters={'deadline':unix_time_converter,
                                'state_changed_at':unix_time_converter,
                                'created_at':unix_time_converter,
                                'launched_at':unix_time_converter})

print('The train data has {} rows and {} columns'.format(train.shape[0],train.shape[1]))
print('The test data has {} rows and {} columns'.format(test.shape[0],test.shape[1]))

#curr = CurrencyRates()
curr2 = CurrencyConverter()
'''
feat = ['disable_communication','country']
for x in feat:
    le = LabelEncoder()
    le.fit(list(train[x].values) + list(test[x].values))
    train[x] = le.transform(list(train[x]))
    test[x] = le.transform(list(test[x]))
'''
#print(train)
#print(test)
#sys.exit(1)
'''
def curr_convert(x):
    try:
        print("Using method 1")
        print(x['currency'])
        return round(curr.convert(base_cur=x['currency'], dest_cur='USD', amount=x['goal'],
                           date_obj=pd.to_datetime(x['created_at'])), 0)
    except:
        try:
            print("Using method 2")
            return round(curr.convert(base_cur=x['currency'], dest_cur='USD', amount=x['goal'],
                                      date_obj=(pd.to_datetime(x['created_at']) + datetime.timedelta(days=1))), 0)
        except:
            print("Using method 3")
            return round(curr.convert(base_cur=x['currency'], dest_cur='USD', amount=x['goal']), 0)
'''


def curr_convert2(x):
    try:
        return round(curr2.convert(currency=x['currency'], new_currency='USD', amount=x['goal'],
                                   date=pd.to_datetime(x['created_at'])), 0)
    except:
        try:
            return round(curr2.convert(currency=x['currency'], new_currency='USD', amount=x['goal'],
                                       date=(pd.to_datetime(x['created_at']) + datetime.timedelta(days=3))), 0)
        except:
            try:
                return round(curr2.convert(currency=x['currency'], new_currency='USD', amount=x['goal'],
                                           date=(pd.to_datetime(x['deadline']))), 0)
            except:
                return round(curr2.convert(currency=x['currency'], new_currency='USD', amount=x['goal']), 0)


#train['conv_goal'] = train.apply(lambda x: round(curr2.convert(currency=x['currency'], new_currency='USD', amount=x['goal'], date=pd.to_datetime(x['created_at'])),0), axis=1)
#train['conv_goal'] = train.apply(lambda x: round(curr.convert(base_cur=x['currency'], dest_cur='USD', amount=x['goal'], date_obj=pd.to_datetime(x['created_at'])),0), axis=1)
train['goal'] = train.apply(lambda x: x['goal'] if x['currency'] == 'USD' else curr_convert2(x), axis=1)
test['goal'] = test.apply(lambda x: x['goal'] if x['currency'] == 'USD' else curr_convert2(x), axis=1)


train['fund_duration_hrs'] = pd.to_timedelta(pd.to_datetime(train['deadline']) - pd.to_datetime(train['launched_at']))
train['fund_duration_hrs'] = train.apply(lambda x: np.ceil(x['fund_duration_hrs'].total_seconds()/3600), axis=1)
train['fund_duration_min'] = pd.to_timedelta(pd.to_datetime(train['deadline']) - pd.to_datetime(train['launched_at']))
train['fund_duration_min'] = train.apply(lambda x: np.ceil(x['fund_duration_min'].total_seconds()/60), axis=1)
train['fund_duration_sec'] = pd.to_timedelta(pd.to_datetime(train['deadline']) - pd.to_datetime(train['launched_at']))
train['fund_duration_sec'] = train.apply(lambda x: np.ceil(x['fund_duration_sec'].total_seconds()), axis=1)

test['fund_duration_hrs'] = pd.to_timedelta(pd.to_datetime(test['deadline']) - pd.to_datetime(test['launched_at']))
test['fund_duration_hrs'] = test.apply(lambda x: np.ceil(x['fund_duration_hrs'].total_seconds()/3600), axis=1)
test['fund_duration_min'] = pd.to_timedelta(pd.to_datetime(test['deadline']) - pd.to_datetime(test['launched_at']))
test['fund_duration_min'] = test.apply(lambda x: np.ceil(x['fund_duration_min'].total_seconds()/60), axis=1)
test['fund_duration_sec'] = pd.to_timedelta(pd.to_datetime(test['deadline']) - pd.to_datetime(test['launched_at']))
test['fund_duration_sec'] = test.apply(lambda x: np.ceil(x['fund_duration_sec'].total_seconds()), axis=1)

train['canceled'] = np.where(pd.to_datetime(train['deadline']) > pd.to_datetime(train['state_changed_at']), True, False)
test['canceled'] = np.where(pd.to_datetime(test['deadline']) > pd.to_datetime(test['state_changed_at']), True, False)

print(train)
print(test)

cols = train.columns.tolist()
cols.insert(len(cols), cols.pop(cols.index('final_status')))
print(cols)
train = train.reindex(columns= cols)

train.to_csv("C:\\Users\\satvi\\Documents\\Projects\\Hackerearth - ML 2\\Data Files\\initial_proc_train.csv",
             index=False)
test.to_csv("C:\\Users\\satvi\\Documents\\Projects\\Hackerearth - ML 2\\Data Files\\initial_proc_test.csv",
            index=False)