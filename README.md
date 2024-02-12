This is final project from Machine Learing 2023.
The requirement is to recommend a model for a DJ house to judge songs' "Danceability", which is an integer in [0, 9]. 
The report shows the approach we did.

## Preprocess

Firstly, we have to do NLP.
For natural language, we split them into 10 dimensions (for Danceability in [0, 9]).
And we have three approaches to analyze them.

### NLP
1. description and title
The data in this type is usually composed of a long paragraph or a long sentence
We can seperate each word and study its frequency

2. composer, artist and channel
The data in this type is usually composed name, with a lot of repitition 
(for extreme case, over 10% of the data has the same artist)
We can study its frequency of appearence

3. album and track
The data in this type is usually composed name, with few repitition (most of them is 1, up to 11)
We can study its frequency of appearence, but we discard the ones with low rate of repitition (<3)

### Imputing

Use KNN after NLP.

## Training

We use some models to analyze it.

1. Linear Regression
2. SVM
3. AdaBoost
4. XGBoost

After tuning, we recommend XGBoost as our result.