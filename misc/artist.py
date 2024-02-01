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
print(dataset) #24

songs = [0] * 10
artists = [[],[],[],[],[],[],[],[],[],[]] 

for i in tqdm(range(len(dataset))) :
    artists[(int)(dataset["Danceability"][i])].append(str(dataset["Composer"][i]))
    songs[(int)(dataset["Danceability"][i])] += 1

artists_arranged = []  

for i in range (10) :
    artists[i] = list(filter(lambda a: a != 'nan', artists[i]))
    artists_arranged.append(FreqDist(artists[i]).most_common(50))
    for j in range (len(artists_arranged[i])) :
        artists_arranged[i][j] = list(artists_arranged[i][j])
        artists_arranged[i][j][1] /= songs[i]
    print( artists_arranged[i] )
