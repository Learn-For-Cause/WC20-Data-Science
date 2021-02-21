import sys
import pandas as pd
import numpy as np
from nltk.stem.snowball import PorterStemmer
import datetime
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import Word2Vec
from collections import Counter

start_time = datetime.datetime.now()

stop = set(stopwords.words("english"))
stemmer = PorterStemmer()


def tokenizer(x):
    words = CountVectorizer().build_tokenizer()(x)
    if not words:
        return []
    ret = [str(stemmer.stem(str(w))) for w in words if w.lower() not in stop]
    return ret

train = pd.read_csv("C:\\Users\\satvi\\Documents\\Projects\\Hackerearth - ML 2\\Data Files\\initial_proc_train.csv",
                    header=0, encoding='latin-1',
                    converters={'desc': tokenizer})

test = pd.read_csv("C:\\Users\\satvi\\Documents\\Projects\\Hackerearth - ML 2\\Data Files\\initial_proc_test.csv",
                    header=0, encoding='latin-1',
                    converters={'desc': tokenizer})


df1 = pd.DataFrame()
df2 = pd.DataFrame()


def calculate_words_ratio(col):
    words_1 = []
    train[train['final_status'] == 1].apply(lambda x: words_1.append(x[col]),axis=1)
    words_1_flat = [item for sub in words_1 for item in sub]
    print(col + " 1 " + str(len(words_1_flat) * 0.04))
    n = int(np.ceil(len(words_1_flat) * 0.04))
    df_words = Counter(words_1_flat).most_common(n)
    words_1_list = [x[0] for x in df_words]
    words_0 = []
    train[train['final_status'] == 0].apply(lambda x: words_0.append(x[col]),axis=1)
    words_0_flat = [item for sub in words_0 for item in sub]
    n = int(np.ceil(len(words_0_flat) * 0.04))
    df_words = Counter(words_0_flat).most_common(n)
    words_0_list = [x[0] for x in df_words]
    unique_words = list(set(words_1_list).difference(set(words_0_list)))
    print(unique_words)

    def helper(x):
        print(x)
        count = 0
        for each in x:
            if each in unique_words:
                count += 1
        return count

    df1['unique_count'] = train.apply(
        lambda x: helper(x[col]) if len(x[col]) != 0 else 0, axis=1)

    df2['unique_count'] = test.apply(
        lambda x: helper(x[col]) if len(x[col]) != 0 else 0, axis=1)

calculate_words_ratio('desc')
#calculate_words_ratio('keywords', 100)
df1.to_csv("C:\\Users\\satvi\\Documents\\Projects\\Hackerearth - ML 2\\Data Files\\count_unique_train.csv", index=False)
df2.to_csv("C:\\Users\\satvi\\Documents\\Projects\\Hackerearth - ML 2\\Data Files\\count_unique_test.csv", index=False)

print("Time taken: " + str(datetime.datetime.now() - start_time))