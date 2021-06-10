#Problem Statement: -
#In this case study, you have been given Twitter data collected from an anonymous twitter handle. With the help of a Naïve Bayes model, predict if a given tweet about a real disaster is real or fake.
#1 = real tweet and 0 = fake tweet

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer

# Loading the data set
twitter_data = pd.read_csv("D:\\360Digi\\naive bayes\\Disaster_tweets_NB.csv")


import re
stop_words = []
# Load the custom built Stopwords
with open("D:/360Digi/Machine learning/Text mining/stopwords_en.txt","r") as sw:
    stop_words = sw.read()

stop_words = stop_words.split("\n")
   
def cleaning_text(i):
    i = re.sub("[^A-Za-z" "]+"," ",i).lower()
    i = re.sub("[0-9" "]+"," ",i)
    w = []
    for word in i.split(" "):
        if len(word)>3:
            w.append(word)
    return (" ".join(w))

# testing above function with sample text => removes punctuations, n

twitter_data.text = twitter_data.text.apply(cleaning_text)
twitter_data.text

# removing empty rows
twitter_data = twitter_data.loc[twitter_data.text != " ",:]

# CountVectorizer
# Convert a collection of text documents to a matrix of token counts

# splitting data into train and test data sets 
from sklearn.model_selection import train_test_split

twitter_train, twitter_test = train_test_split(twitter_data, test_size = 0.2)

# creating a matrix of token counts for the entire text document 
def split_into_words(i):
    return [word for word in i.split(" ")]

# Defining the preparation of twiiter texts into word count matrix format - Bag of Words
twitter_bow = CountVectorizer(analyzer = split_into_words).fit(twitter_data.text)

# Defining BOW for all messages
all_twitter_matrix = twitter_bow.transform(twitter_data.text)

# For training messages
train_twitter_matrix = twitter_bow.transform(twitter_train.text)

# For testing messages
test_twitter_matrix = twitter_bow.transform(twitter_test.text)

# Learning Term weighting and normalizing on entire emails
tfidf_transformer = TfidfTransformer().fit(all_twitter_matrix)

# Preparing TFIDF for train emails
train_tfidf = tfidf_transformer.transform(train_twitter_matrix)
train_tfidf.shape # (row, column)

# Preparing TFIDF for test emails
test_tfidf = tfidf_transformer.transform(test_twitter_matrix)
test_tfidf.shape #  (row, column)


# Preparing a naive bayes model on training data set 

from sklearn.naive_bayes import MultinomialNB as MB

# Multinomial Naive Bayes
classifier_mb = MB()
classifier_mb.fit(train_tfidf, twitter_train.target)

# Evaluation on Test Data
test_pred_m = classifier_mb.predict(test_tfidf)
test_pred_m
accuracy_test_m = np.mean(test_pred_m == twitter_test.target)
accuracy_test_m

from sklearn.metrics import accuracy_score
accuracy_score(test_pred_m, twitter_test.target) 

pd.crosstab(test_pred_m, twitter_test.target)

# Training Data accuracy
train_pred_m = classifier_mb.predict(train_tfidf)
accuracy_train_m = np.mean(train_pred_m == twitter_train.target)
accuracy_train_m


print("accuracy_test", accuracy_test_m)
print("Crosstab",pd.crosstab(test_pred_m, twitter_test.target))
print("accuracy_train",accuracy_train_m)

# Multinomial Naive Bayes changing default alpha for laplace smoothing
# if alpha = 0 then no smoothing is applied and the default alpha parameter is 1
# the smoothing process mainly solves the emergence of zero probability problem in the dataset.



classifier_mb_lap = MB(alpha = 13)
classifier_mb_lap.fit(train_tfidf, twitter_train.target)

# Evaluation on Test Data after applying laplace
test_pred_lap = classifier_mb_lap.predict(test_tfidf)
accuracy_test_lap = np.mean(test_pred_lap == twitter_test.target)
accuracy_test_lap

from sklearn.metrics import accuracy_score
accuracy_score(test_pred_lap, twitter_test.target) 

pd.crosstab(test_pred_lap, twitter_test.target)

# Training Data accuracy
train_pred_lap = classifier_mb_lap.predict(train_tfidf)
accuracy_train_lap = np.mean(train_pred_lap == twitter_train.target)
accuracy_train_lap

print("accuracy_test", accuracy_test_lap)
print("Crosstab",pd.crosstab(test_pred_lap, twitter_test.target))
print("accuracy_train",accuracy_train_lap)



