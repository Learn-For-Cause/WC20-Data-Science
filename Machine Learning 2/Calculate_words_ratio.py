import sys
import pandas as pd
import numpy as np
from nltk.stem.snowball import EnglishStemmer
import nltk
import datetime
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
from gensim.models import doc2vec

start_time = datetime.datetime.now()

stop = set(stopwords.words("english"))
stemmer = EnglishStemmer()

# ------------------------------------------------- Helper Function for tokenizing/stemming string


def tokenizer(x):
    words = CountVectorizer().build_tokenizer()(x)
    ret = [stemmer.stem(w) for w in words if w.lower() not in stop]
    return list(set(ret))


train = pd.read_csv("C:\\Users\\satvi\\Documents\\Projects\\Hackerearth - ML 2\\Data Files\\initial_proc_train.csv",
                    header=0, encoding='latin-1',
                    converters={'desc': tokenizer, 'keywords': tokenizer})

test = pd.read_csv("C:\\Users\\satvi\\Documents\\Projects\\Hackerearth - ML 2\\Data Files\\initial_proc_test.csv",
                    header=0, encoding='latin-1',
                    converters={'desc': tokenizer, 'keywords': tokenizer})
# , 'keywords': tokenizer

df1 = pd.DataFrame()
df2 = pd.DataFrame()


def calculate_words_ratio(col, n):
    words_1 = []
    train[train['final_status'] == 1].apply(lambda x: words_1.append(x[col]),axis=1)
    words_1_flat = [item for sub in words_1 for item in sub]
    print(col + " 1 " + str(len(words_1_flat)))
    df_words = Counter(words_1_flat).most_common(n)
    words_1_list = [x[0] for x in df_words]
    df1[col[0:1]+'_count_1'] = train.apply(
        lambda x: len(set(words_1_list).intersection(x[col])) / len(x[col]) if len(x[col]) != 0 else 0, axis=1)

    words_0 = []
    train[train['final_status'] == 0].apply(lambda x: words_0.append(x[col]),axis=1)
    words_0_flat = [item for sub in words_0 for item in sub]
    print(col + " 0 " + str(len(words_0_flat)))
    df_words = Counter(words_0_flat).most_common(n)
    words_0_list = [x[0] for x in df_words]
    df1[col[0:1]+'_count_0'] = train.apply(
        lambda x: len(set(words_0_list).intersection(x[col])) / len(x[col]) if len(x[col]) != 0 else 0, axis=1)

    df2[col[0:1]+'_count_1'] = test.apply(
        lambda x: len(set(words_1_list).intersection(x[col])) / len(x[col]) if len(x[col]) != 0 else 0, axis=1)
    df2[col[0:1]+'_count_0'] = test.apply(
        lambda x: len(set(words_0_list).intersection(x[col])) / len(x[col]) if len(x[col]) != 0 else 0, axis=1)

calculate_words_ratio('desc', 500)
#calculate_words_ratio('keywords', 100)
df1.to_csv("C:\\Users\\satvi\\Documents\\Projects\\Hackerearth - ML 2\\Data Files\\count_key_train.csv", index=False)
df2.to_csv("C:\\Users\\satvi\\Documents\\Projects\\Hackerearth - ML 2\\Data Files\\count_key_test.csv", index=False)

print("Time taken: " + str(datetime.datetime.now() - start_time))

