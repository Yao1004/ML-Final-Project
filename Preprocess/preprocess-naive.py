import pandas as pd
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.tokenize import RegexpTokenizer
from numpy.ma.extras import median
import statistics

pd.options.mode.chained_assignment = None

# The following data is seperated into three types:
#
# Type1: description and title
# The data in this type is usually composed of a long paragraph or a long sentence
# We can seperate each word and study its frequency
# 
# Type2: composer, artist and channel
# The data in this type is usually composed name, with a lot of repitition 
# (for extreme case, over 10% of the data has the same artist)
# We can study its frequency of appearence
#
# Type3: album and track
# The data in this type is usually composed name, with few repitition (most of them is 1, up to 11)
# We can study its frequency of appearence, but we discard the ones with low rate of repitition (<3)

TYPE = 0
ORIG = 1
RESULT = 2
SIZE = 3
AVG = 4
AVG_ALL = 5

nl = { # type, orig, result, size_to_fills, average due to danceability, average for all
    'Description' : [ 1, [''] * 10              , [] , 100 , [[0 for i in range(10)] for j in range(10)] , [0]*10 ],
    'Title' :       [ 1, [''] * 10              , [] , 50  , [[0 for i in range(10)] for j in range(10)] , [0]*10 ],
    'Composer' :    [ 2, [[] for i in range(10)], [] , 50  , [[0 for i in range(10)] for j in range(10)] , [0]*10 ],
    'Artist' :      [ 2, [[] for i in range(10)], [] , 50  , [[0 for i in range(10)] for j in range(10)] , [0]*10 ],
    'Channel' :     [ 2, [[] for i in range(10)], [] , 50  , [[0 for i in range(10)] for j in range(10)] , [0]*10 ],
    'Album' :       [ 3, [[] for i in range(10)], [] , 50  , [[0 for i in range(10)] for j in range(10)] , [0]*10 ],
    'Track' :       [ 3, [[] for i in range(10)], [] , 50  , [[0 for i in range(10)] for j in range(10)] , [0]*10 ],
}

#inname = input('the train input filename : ')
#outname = input('the train output filename : ')
inname = 'train.csv'
outname = 'modified.csv'
dataset = pd.read_csv(inname,encoding='utf-8')

songs = [0] * 10 # the number of songs in each danceability
length_train = len(dataset['Danceability'])

# Transform to a number
def quantizer(text, set, type) :
    if type == 1 :
        val = 0.0
        for i in set :
            if i[0] in text :
                val += i[1]
        return val
    else :
        for i in set :
            if i[0] == text :   
                return i[1]
        return 0
    

# For type 1, we seperate it into 10 huge strings
# For type 2 and 3, we put it in the list  
for i in tqdm(range(len(dataset))) :
    for key in nl :
        if nl[key][TYPE] == 1:
            nl[key][ORIG][(int)(dataset["Danceability"][i])] += ' ' + str(dataset[key][i])
        else :
            nl[key][ORIG][(int)(dataset["Danceability"][i])].append(str(dataset[key][i]))
    songs[(int)(dataset["Danceability"][i])] += 1

# Stop words means the word that wouldn't affect the meaning, for example, 'I', 'you','the' , are some of those
# We add some extra words that seems with weak relation with danceability
stopword = stopwords.words('english') + stopwords.words('spanish')
stopword += ['https','http','com','www','nan','youtube','music']
tokenizer = RegexpTokenizer(r'\w+|\'\w+', gaps = False)

punct_description = [[] for i in range(10)]
punct_title = [[] for i in range(10)]

# 1.Remove words that doesn't belong to utf-8
# 2.Turn all the alphabets into lower cases
# 3.Tokenize the words
for i in range(10) :
    nl['Description'][ORIG][i] = nl['Description'][ORIG][i].encode('utf-8', errors='ignore').decode('utf-8')
    nl['Description'][ORIG][i] = nl['Description'][ORIG][i].lower()
    punct_description[i] = tokenizer.tokenize(nl['Description'][ORIG][i])

    nl['Title'][ORIG][i] = nl['Title'][ORIG][i].encode('utf-8', errors='ignore').decode('utf-8')
    nl['Title'][ORIG][i] = nl['Title'][ORIG][i].lower()
    punct_title[i] = tokenizer.tokenize(nl['Title'][ORIG][i])

tokenized_description = [[] for i in range(10)]
tokenized_title = [[] for i in range(10)]

for i in range(10) :
    for j in punct_description[i] :
        if j not in stopword :
            tokenized_description[i].append(j)
    for j in punct_title[i] :
        if j not in stopword :
            tokenized_title[i].append(j)
    
    # Remove the NIL values
    for key in nl :
        if nl[key][TYPE] != 1 :
            nl[key][ORIG][i] = list(filter(lambda a: a.lower() != 'nan', nl[key][ORIG][i]))

    # FreqDist is a convenient tool to count the most common words
    nl['Description'][RESULT].append(FreqDist(tokenized_description[i]).most_common(nl['Description'][SIZE]))
    nl['Title'][RESULT].append(FreqDist(tokenized_description[i]).most_common(nl['Title'][SIZE]))
    for key in nl :
        if nl[key][TYPE] != 1 :
            nl[key][RESULT].append(FreqDist(nl[key][ORIG][i]).most_common(nl[key][SIZE]))

    # We divide the number of songs in each danceability, showing the frequency 
    for key in nl :
        if nl[key][TYPE] != 3 :
            for j in range(len(nl[key][RESULT][i])) :
                nl[key][RESULT][i][j] = list(nl[key][RESULT][i][j])
                nl[key][RESULT][i][j][1] /= songs[i]
        else :
            nl[key][RESULT][i] = list(filter(lambda a: a[1] >= 3, nl[key][RESULT][i]))
            for j in range (len(nl[key][RESULT][i])) :
                nl[key][RESULT][i][j] = list(nl[key][RESULT][i][j])
                nl[key][RESULT][i][j][1] /= songs[i]

# For those with NIL, we give it a average value
# avg[i][j] means ones with dancibility i, with analysis of j-th , in average has the value of this.
for key in nl :
    for i in range (len(dataset[key])) :
        if str(dataset[key][i]).lower() != 'nan' :
            for j in range(10) :
                nl[key][AVG][int(dataset['Danceability'][i])][j] += quantizer(dataset[key][i],nl[key][RESULT][j],nl[key][TYPE])
                nl[key][AVG_ALL][j] += quantizer(dataset[key][i],nl[key][RESULT][j],nl[key][TYPE])
    for i in range (10) :
        nl[key][AVG][i] = [ x / songs[i] for x in nl[key][AVG][i] ]
        nl[key][AVG_ALL] = [x / length_train for x in nl[key][AVG_ALL] ]
    
for key in nl :
    for i in range(10) :
        t = []
        for j in range(len(dataset[key])) :
            if str(dataset[key][j]).lower() != 'nan' :
                t.append(quantizer(dataset[key][j],nl[key][RESULT][i],nl[key][TYPE]))
            else :
                t.append( nl[key][AVG][i][(int)(dataset['Danceability'][j])] )
        dataset[ key + ' of ' + str(i)] = t

# fill the rest
to_fill = {  # type, avg/median due to danceability, avg/median due to all
    'Energy':               [1, [0] * 10, 0],
    'Loudness':             [0, [0] * 10, 0],
    'Speechiness' :         [0, [0] * 10, 0],
    'Acousticness' :        [0, [0] * 10, 0],
    'Instrumentalness' :    [1, [0] * 10, 0],
    'Liveness' :            [1, [0] * 10, 0],
    'Valence' :             [1, [0] * 10, 0],
    'Tempo' :               [0, [0] * 10, 0],
    'Duration_ms' :         [1, [0] * 10, 0],
    'Views' :               [1, [0] * 10, 0],
    'Likes' :               [1, [0] * 10, 0],
    'Stream' :              [1, [0] * 10, 0],
    'Comments' :            [1, [0] * 10, 0] 
}

for key in to_fill :
    for dance_val in range(10) :
        temp_list = []
        for i in range(length_train) :
            if int(dataset['Danceability'][i]) == dance_val and str(dataset[key][i]) != "nan" :
                temp_list.append(dataset[key][i])
        if to_fill[key][0] == 0 :
            to_fill[key][1][dance_val] = sum(temp_list) / len(temp_list)
        else :
            to_fill[key][1][dance_val] = statistics.median(temp_list)

for key in to_fill :
    for i in range(length_train) :
        if str(dataset[key][i]).lower() == "nan" :
            dataset[key][i] = to_fill[key][1][int(dataset['Danceability'][i])]

# avg/median for all
for key in to_fill :
    temp_list = []
    for i in range(length_train) :
        if str(dataset[key][i]) != "nan" :
            temp_list.append(dataset[key][i])
    if to_fill[key][0] == 0 :
        to_fill[key][2] = sum(temp_list)/len(temp_list)
    else :
        to_fill[key][2] = statistics.median(temp_list)

avg_key = [[0 for i in range(11)] for j in range(10)]

for i in range(11) :
    dataset['Key of ' + str(i)] = [0]*length_train

# split key into 11 dimension
for i in range(length_train) :
    if str(dataset['Key'][i]).lower() != 'nan' and str(dataset['Key'][i]).lower() != '-1' :
        avg_key[int(dataset['Danceability'][i])][int(dataset['Key'][i])] += 1

for i in range(10) :
    avg_key[i] = [ x / songs[i] for x in avg_key[i] ]

for i in range(length_train) :
    if str(dataset['Key'][i]).lower() != 'nan' :
        for j in range(11) :
            dataset['Key of ' + str(int(dataset['Key'][i]))][i] = 1
    else :
        for k in range(11) :
            dataset['Key of ' + str(k)][i] = avg_key[int(dataset['Danceability'][i])][k]

# avg for all
avg_key_test = [0]*11
for i in range(length_train) :
    if str(dataset['Key'][i]).lower() != 'nan' :
        avg_key_test[int(dataset['Key'][i])] += 1
avg_key_test = [ x / length_train for x in avg_key_test ]
    
del dataset['Description'],dataset['Title'],dataset['Composer']
del dataset['Artist'],dataset['Channel'],dataset['Album'],dataset['Track']
del dataset['id'], dataset['Uri'], dataset['Url_spotify'], dataset['Url_youtube']
del dataset['Album_type'], dataset['Licensed'], dataset['official_video'],dataset['Key'] # Album_type, Licensed, official_video not yet dealed

pd.DataFrame(dataset).to_csv(outname, index=False)

# Finish dealing with train 
# Now start dealing with test

#inname2 = input('the test input filename : ') 
#outname2 = input('the test output filename : ')
inname2 = 'test.csv'
outname2 = 'modified_test.csv'
testdata = pd.read_csv(inname2,encoding = 'utf-8')
length_test = len(testdata['Key'])

# fill the data with average/median

for key in to_fill :
    for i in range(length_test) :
        if str(testdata[key][i]).lower() == "nan" :
            testdata[key][i] = to_fill[key][2]

# convert the natural language into number
for key in nl :
    for i in range (10) :
        testdata[ key + ' of ' + str(i) ] = [0]*length_test

for i in tqdm(range(len(testdata))) :
    for key in nl :
        if str(testdata[key][i]).lower != 'nan' and str(testdata[key][i]).lower != '-1':
            for j in range (10) :
                testdata[ key + ' of ' + str(j) ][i] = quantizer(str(testdata[key][i]), nl[key][RESULT][j], nl[key][TYPE])
        else :
            for j in range (10) :
                testdata[ key + ' of ' + str(j) ][i] = nl[key][AVG_ALL][j]

# Turn key into 11-dim data
for i in range(11) :
    testdata['Key of ' + str(i)] = [0] * length_test
for i in range(length_test) :
    if str(testdata['Key'][i]).lower() != 'nan' :
        for j in range(11) :
            testdata['Key of ' + str(int(testdata['Key'][i]))][i] = 1
    else :
        for k in range(11) :
            testdata['Key of ' + str(k)][i] = avg_key_test[k]

del testdata['Description'],testdata['Title'],testdata['Composer']
del testdata['Artist'],testdata['Channel'],testdata['Album'],testdata['Track']
del testdata['Uri'], testdata['Url_spotify'], testdata['Url_youtube']
del testdata['Album_type'], testdata['Licensed'], testdata['official_video'],testdata['Key'],testdata['id']

pd.DataFrame(testdata).to_csv(outname2, index=False)
