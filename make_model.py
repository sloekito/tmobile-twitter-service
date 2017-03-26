import random
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import sklearn.model_selection as ms
import re
import scipy
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

class TwitterClassifications(object):
    def __init__(self):
        floc = './data/pos_twitter.json'
        with open(floc , 'r') as fhand:
            tweets = json.load(fhand)
        with open(floc , 'r') as fhand:
            tweets = json.load(fhand)
        text = []
        for row in tweets:
            row1 = row['result']['extractorData']['data']
            for subrow in row1:
                row_sub2 = subrow['group']
                for sub2row in row_sub2:
                    row_sub3 = sub2row['Tweettextsizeblock']
                    for sub3row in row_sub3:
                        text.append(sub3row['text'])
        good_text = []
        for tweet in text:
            add_words = []
            words = tweet.split()
            for word in words:
                word = word.lower().strip()
                if re.search('^[a-z#]', word.lower()) and '/' not in word:
                    if len(re.sub('[^a-z]', '', word)) > 3:
                        add_words.append(re.sub('[^a-z]', '', word))
            good_text.append(' '.join(add_words))
        ### neg text data extraction
        bloc = './data/neg_twitter{}.txt'
        with open(bloc.format('')) as fhand:
            texty = fhand.read()
        with open(bloc.format('_2')) as fhand:
            texty += fhand.read()
        with open(bloc.format('_3')) as fhand:
            textish = fhand.read()
        with open(bloc.format('_4')) as fhand:
            textish += fhand.read()
        badtweets = textish.split('||||||||||||||||||||||')
        badtweets += texty.split('\n')
        bad_text = []
        for tweet in badtweets:
            add_words = []
            words = tweet.split()
            for word in words:
                word = word.lower().strip()
                if re.search('^[a-z#]', word.lower()) and '/' not in word:
                    if len(re.sub('[^a-z]', '', word)) > 3:
                        add_words.append(re.sub('[^a-z]', '', word))
            bad_text.append(' '.join(add_words))
        random.shuffle(good_text)
        random.shuffle(bad_text)
        ### build balanced classes
        if len(good_text) > len(bad_text):
            good_text = good_text[:len(bad_text)]
        else:
            bad_text = bad_text[:len(good_text)] 
        good_to_bad = len(good_text)
        ### add labels
        data = []
        for line in good_text:
            data.append({'text' : line, 'label': 1})
        for line in bad_text:
            data.append({'text' : line, 'label': 0})
        random.shuffle(data)
        all_tweets = pd.DataFrame(data)
        ys = all_tweets.pop('label')
        self.corpus = all_tweets
        self.labels = ys

    def train_model(self):
        X_train = self.corpus['text'].values
        y_train = self.labels
        classifier = Pipeline([
            ('vectorizer', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', OneVsRestClassifier(RandomForestClassifier(n_jobs=-1, max_depth=20,
                                                            n_estimators=50)))])
        self.classifier = classifier.fit(X_train, y_train)

    def predict(self, input):
        predicted = self.classifier.predict_proba(input)[:,1]
        return predicted.mean() * 100