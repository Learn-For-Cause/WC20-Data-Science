import sys
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from nltk.stem.snowball import EnglishStemmer
import nltk
import datetime
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter

'''
Extracting top words from 'keywords' column and generating one hot encoding of the same
'''

# ------------------------------------------------- Initializing

start_time = datetime.datetime.now()

stop = set(stopwords.words("english"))
stemmer = EnglishStemmer()

# ------------------------------------------------- Helper Function for tokenizing/stemming string


def tokenizer(x):
    words = CountVectorizer().build_tokenizer()(x)
    ret = [stemmer.stem(w) for w in words if w.lower() not in stop]
    return list(set(ret))

# ------------------------------------------------- Reading files

test = pd.read_csv("C:\\Users\\satvi\\Documents\\Projects\\Hackerearth - ML 2\\Data Files\\initial_proc_test.csv",
                    header=0, usecols=range(4,5), encoding='latin-1',
                    converters={'keywords': tokenizer})

train = pd.read_csv("C:\\Users\\satvi\\Documents\\Projects\\Hackerearth - ML 2\\Data Files\\initial_proc_train.csv",
                    header=0, usecols=range(4,5), encoding='latin-1',
                    converters={'keywords': tokenizer})

# ------------------------------------------------- Helper function to extract top words from test & train

def word_generator(df1, df2, type):
    df_words = Counter(" ".join(df1["keywords"].str.join(' ')).split()).most_common(100)
    print(df_words)
    df_words_list = [x[0] for x in df_words]
    print(df_words_list)

    def one_hot_encoding(df, word_list, type2):
        tw = pd.DataFrame()
        for word in word_list:
            print(word)
            tw["k_"+ word] = df.apply(lambda x: 1 if word in x['keywords'] else 0, axis=1)
        print(tw.shape)
        path = "C:\\Users\\satvi\\Documents\\Projects\\Hackerearth - ML 2\\Data Files\\"+ str(type) + "_keywords_in_" + str(type2) + ".csv"
        tw.to_csv(path, index=False)

    if type == "test":
        one_hot_encoding(df1, df_words_list, "test")
        one_hot_encoding(df2, df_words_list, "train")
    else:
        one_hot_encoding(df1, df_words_list, "train")
        one_hot_encoding(df2, df_words_list, "test")

word_generator(test, train, "test")
word_generator(train, test, "train")
print("Time taken: " + str(datetime.datetime.now() - start_time))

