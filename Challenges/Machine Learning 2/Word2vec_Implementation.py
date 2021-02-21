import sys
import pandas as pd
import numpy as np
from nltk.stem.snowball import PorterStemmer
import datetime
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import Word2Vec
#from gensim.parsing import PorterStemmer

start_time = datetime.datetime.now()

stop = set(stopwords.words("english"))
stemmer = PorterStemmer()
words_list = []

# ------------------------------------------------- Helper Function for tokenizing/stemming string


def tokenizer(x):
    words = CountVectorizer().build_tokenizer()(x)
    if not words:
        return []
    ret = [str(stemmer.stem(str(w))) for w in words if w.lower() not in stop]
    words_list.append(ret)
    return ret

train = pd.read_csv("C:\\Users\\satvi\\Documents\\Projects\\Hackerearth - ML 2\\Data Files\\initial_proc_train.csv",
                    header=0, encoding='latin-1',
                    converters={'desc': tokenizer})

test = pd.read_csv("C:\\Users\\satvi\\Documents\\Projects\\Hackerearth - ML 2\\Data Files\\initial_proc_test.csv",
                    header=0, encoding='latin-1',
                    converters={'desc': tokenizer})

size = 20
w2v_model = Word2Vec(size=size, min_count=1, iter=30, seed=69, workers=3)
w2v_model.build_vocab(words_list)


def build_vec(x):
    frame = []
    for row in x['desc']:
        vec = [0] * size
        if row:
            for word in row:
                vec += w2v_model.wv[word]
        frame.append(vec)
    return frame

columns = ["w2v_col_"+str(i) for i in range(0,size)]
df1 = pd.DataFrame(build_vec(train), columns=columns)
df2 = pd.DataFrame(build_vec(test), columns=columns)
df1.to_csv("Data Files\\W2V_train.csv", index=False)
df2.to_csv("Data Files\\W2V_test.csv", index=False)

print("Time taken: " + str(datetime.datetime.now() - start_time))