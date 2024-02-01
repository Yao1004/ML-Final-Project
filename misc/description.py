import nltk
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.tokenize import RegexpTokenizer

dataset = pd.read_csv("train.csv")
# sorted(datalist, key = lambda s:s[0])
# print(dataset["Description"][0]) #24

strings = [''] * 10
songs = [0] * 10
for i in tqdm(range(len(dataset))) :
    strings[(int)(dataset["Danceability"][i])] += ' ' + str(dataset["Description"][i])
    songs[(int)(dataset["Danceability"][i])] += 1

stopword = stopwords.words('english') + stopwords.words('spanish')
stopword += ['https','http','com','www','nan']
tokenizer = RegexpTokenizer(r'\w+|\'\w+', gaps = False)

punct = [[],[],[],[],[],[],[],[],[],[]]
tokenized = [[],[],[],[],[],[],[],[],[],[]]

for i in range( len(strings) ) :
    strings[i].encode('utf-8', errors='ignore').decode('utf-8')
    strings[i] = strings[i].lower()
    punct[i] = tokenizer.tokenize(strings[i])

dict_words = []

for i in range( len(punct) ) :
    for j in punct[i] :
        if j not in stopword :
            tokenized[i].append(j)
    dict_words.append(FreqDist(tokenized[i]).most_common(50))
    for j in range (50) :
        dict_words[i][j] = list(dict_words[i][j])
        dict_words[i][j][1] /= songs[i]
    print( dict_words[i] )

