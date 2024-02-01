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

songs = [0] * 10
artists = [[],[],[],[],[],[],[],[],[],[]] 

for i in tqdm(range(len(dataset))) :
    if (str(dataset["Track"][i]) != 'nan') :
        artists[(int)(dataset["Danceability"][i])].append(str(dataset["Track"][i]))
        songs[(int)(dataset["Danceability"][i])] += 1

artists_arranged = []  
artists_cleaned = [[],[],[],[],[],[],[],[],[],[]] 

for i in range (10) :
    artists_arranged.append(FreqDist(artists[i]).most_common(50))
    for j in range (50) :
        artists_arranged[i][j] = list(artists_arranged[i][j])
        if(artists_arranged[i][j][1] >= 3) :
            artists_cleaned[i].append(artists_arranged[i][j])
    print( artists_cleaned[i] )
