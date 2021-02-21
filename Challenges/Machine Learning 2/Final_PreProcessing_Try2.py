import numpy as np
import pandas as pd
import sys
from sklearn.preprocessing import LabelEncoder
from nltk.stem.snowball import EnglishStemmer
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

# -------------- Final Pre Processing of train & test data set - TRY 2
# - Tokenizing name, desc, keywords.
# - Stemming the words & removing stopwords
# - Replacing attributes name, desc and keywords with their word counts
# - Binarization of attributes
# - Dropping all time stamps
# - Dropping currency, backers_count & backer_ratio
# --------------

#nltk.download("stopwords")
stop = set(stopwords.words("english"))
stemmer = EnglishStemmer()


'''def tokenizer(x):
    words = CountVectorizer().build_tokenizer()(x)
    ret = [stemmer.stem(w) for w in words if w.lower() not in stop]
    return ret
'''


def tokenizer(x):
    return CountVectorizer().build_tokenizer()(x)

train = pd.read_csv("C:\\Users\\satvi\\Documents\\Projects\\Hackerearth - ML 2\\Data Files\\initial_proc_train.csv",
                    header=0, usecols=range(1,17), encoding='latin-1',
                    converters={'name': tokenizer,
                                'desc': tokenizer,
                                'keywords': tokenizer})

test = pd.read_csv("C:\\Users\\satvi\\Documents\\Projects\\Hackerearth - ML 2\\Data Files\\initial_proc_test.csv",
                    header=0, usecols=range(1,14), encoding='latin-1',
                    converters={'name': tokenizer,
                                'desc': tokenizer,
                                'keywords': tokenizer})

print(train)

len_cols = ['name_len', 'desc_len', 'key_len']
cols = ['name','desc','keywords']
words_col = ['name_words', 'desc_words', 'key_len']

for i in range(0,3):
    train[words_col[i]] = train.apply(lambda x: len(x[cols[i]]), axis=1)
    test[words_col[i]] = test.apply(lambda x: len(x[cols[i]]), axis=1)

train['launched_month'] = pd.to_datetime(train['launched_at']).dt.month
train['deadline_month'] = pd.to_datetime(train['deadline']).dt.month
train['launched_day'] = pd.to_datetime(train['launched_at']).dt.day
train['deadline_day'] = pd.to_datetime(train['deadline']).dt.day
test['launched_month'] = pd.to_datetime(test['launched_at']).dt.month
test['deadline_month'] = pd.to_datetime(test['deadline']).dt.month
test['launched_day'] = pd.to_datetime(test['launched_at']).dt.day
test['deadline_day'] = pd.to_datetime(test['deadline']).dt.day

train.drop(['currency', 'deadline', 'state_changed_at', 'created_at', 'launched_at',
            'backers_count'],
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

train.to_csv("C:\\Users\\satvi\\Documents\\Projects\\Hackerearth - ML 2\\Data Files\\final_proc_train_try2.csv",
             index=False)
test.to_csv("C:\\Users\\satvi\\Documents\\Projects\\Hackerearth - ML 2\\Data Files\\final_proc_test_try2.csv",
            index=False)