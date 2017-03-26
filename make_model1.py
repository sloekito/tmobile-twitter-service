import random
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import sklearn.model_selection as ms
import re
import scipy
import sklearn.linear_model as lm
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from collections import OrderedDict

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
        text = []
        for row in tweets:
            row1 = row['result']['extractorData']['data']
            for subrow in row1:
                row_sub2 = subrow['group']
                for sub2row in row_sub2:
                    row_sub3 = sub2row['Tweettextsizeblock']
                    for sub3row in row_sub3:
                        text.append(sub3row['text'])
        bad_text = []
        for tweet in text:
            add_words = []
            words = tweet.split()
            for word in words:
                word = word.lower().strip()
                if re.search('^[a-z#]', word.lower()) and '/' not in word:
                    if len(re.sub('[^a-z]', '', word)) > 3:
                        add_words.append(re.sub('[^a-z]', '', word))
            bad_text.append(' '.join(add_words))
        ### shuffle the things
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
        return np.sqrt(predicted.mean())

class Financial_Classification(object):
    def __init__(self):
        '''
        loads and cleans the dataset stored in the data folder of this repo
        '''
        # loans = pd.read_csv('s3://gal-deason-aws/loan.csv')
        loans = pd.read_csv('./loan.csv')
        dropcols = ['id', 'member_id', 'funded_amnt_inv', 'grade', 'sub_grade', 'verification_status',
            'url', 'desc', 'title' , 'addr_state', 'dti', 'earliest_cr_line', 'inq_last_6mths',
                'mths_since_last_delinq', 'mths_since_last_record', 'open_acc', 'pub_rec', 'revol_bal'
            'revol_bal', 'revol_util', 'total_acc', 'total_pymnt', 'total_pymnt_inv', 'total_pymnt',
            'total_pymnt_inv','total_rec_prncp', 'total_rec_int','total_rec_late_fee', 'recoveries',
                'collection_recovery_fee','last_pymnt_d','last_pymnt_amnt','next_pymnt_d',
                'last_credit_pull_d','collections_12_mths_ex_med','mths_since_last_major_derog','policy_code','annual_inc_joint'
                ,'dti_joint','verification_status_joint','acc_now_delinq','tot_coll_amt','tot_cur_bal'
                ,'open_acc_6m','open_il_6m','open_il_12m','open_il_24m','mths_since_rcnt_il','total_bal_il'
                ,'il_util','open_rv_12m','open_rv_24m','max_bal_bc','all_util','total_rev_hi_lim'
                ,'inq_fi','total_cu_tl','inq_last_12m'
                    ]
        dropcols = set(dropcols)
        keep_cols = []
        for col in loans.columns:
            if col not in dropcols:
                keep_cols.append(col)
        y_col = 'delinq_2yrs'
        # 'delinq_2yrs' The number of 30+ days past-due incidences of delinquency in the borrower's credit file for the past 2 years
        purpose_keeprows = ['small_business', 'credit_card', 'other']
        loans_ = loans[keep_cols]
        loans_clean = loans_[loans_['purpose'] == 'credit_card']
        loans_clean.dropna(inplace=True)
        loans_clean_ = loans_clean.copy()
        loans_clean_['issue_d'] = loans_clean['issue_d'].apply(pd.Timestamp)
        loans_clean_['year'] = loans_clean_['issue_d'].apply(lambda x: x.year)
        loans_clean_['zip_code'] = loans_clean_['zip_code'].astype('category')
        loans_clean_.drop(['application_type', 'purpose', 'initial_list_status', 'revol_bal',
                        'out_prncp_inv', 'revol_bal', 'pymnt_plan', 'issue_d', 'loan_status'], axis=1, inplace=True)
        rows_need_dummies = ['emp_title', 'home_ownership', 'emp_length', 'term', 'zip_code']
        loans_model = self._make_dummy(loans_clean_, rows_need_dummies)
        neg = loans_model[loans_model[y_col] > 0]
        pos = loans_model[loans_model[y_col] == 0]
        if neg.shape[0] > pos.shape[0]:
            neg = neg.head(pos.shape[0])
        else:
            pos = pos.head(neg.shape[0])
        loans_model = pd.concat([neg, pos])
        loans_y = loans_model.pop(y_col)
        self.data = loans_model
        self.y =  loans_y.apply(lambda x: 1 if x > 0 else 0)
        self.train_model()

    def _make_dummy(self, df, cols):
        df_ = df.copy()
        for col in cols:
            for value in df[col].value_counts().nlargest(50).index:
                df_['{}_{}'.format(col, value)] = df[col] == value
            df_.drop(col, axis=1, inplace=True)
        return df_
    
    def train_model(self):
        '''
        '''
        self.model = lm.LogisticRegression(n_jobs=-1)
        self.model.fit(self.data, self.y)

    def predict(self, title, income):
        '''
        '''
        arguements = OrderedDict()
        for col in self.data.columns:
            arguements[col] = 0
        arguements['loan_amnt'] = 800
        arguements['funded_amnt'] = 800
        arguements['int_rate'] = .05
        arguements['installment'] = 10.8
        arguements['out_prncp'] = self.data['out_prncp'].mean()
        arguements['year'] = 2015
        arguements['emp_title_{}'.format(title)] = 1
        arguements['home_ownership_{}'.format('own'.upper())] = 1
        arguements['emp_length_{}'.format('10+ years')] = 1
        arguements['term_ {}'.format('36 months'.lower())] = 1
        arguements['zip_code_{}xx'.format('980'[:3])] = 1
        aset = set(arguements.keys())
        arguements = pd.DataFrame(arguements.values(), index=arguements.keys()).T
        return self.model.predict_proba(arguements)[:,1][0]

        
    
