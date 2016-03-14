import re
import csv
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn import cross_validation


def clean(string):
  string = re.sub(r"\\t", " ", string)   
  string = re.sub(r"\\n", " ", string)   
  string = re.sub(r"\\r", " ", string)   
  string = re.sub(r"[^A-Za-z0-9\']", " ", string)   
  string = re.sub(r"\s{2,}", " ", string)
  return string.strip()

# load the data
print 'Loading projects.csv...'
projects_df = pd.read_csv('projects.csv')

print 'Loading essays.csv...'
essays_df = pd.read_csv('essays.csv')
essays_df = essays_df.fillna("")

print 'Loading outcomes.csv...'
outcomes_df = pd.read_csv('outcomes.csv')

print('Merging csv files...')
# df is a left outer join of projects and outcomes on projectid
df = pd.merge(projects_df, essays_df, how='left', on='projectid')
df = pd.merge(df, outcomes_df, how='left', on='projectid')

print('Splitting data...')
# split the training data and testing data
dates = np.array(df.date_posted)
test_idx = np.where(dates >= '2014-01-01')[0]
val_idx = np.where(dates < '2014-01-01')[0]
train_idx = np.where(dates < '2013-01-01')[0]
val_idx = np.setdiff1d(val_idx, train_idx)

df.title = df.title.apply(clean)

X = np.array(df.title)
y = np.array(df.is_exciting)

xTr = X[train_idx]
yTr = y[train_idx]
xVal = X[val_idx]
yVal = y[val_idx]

xTe = X[test_idx]

print 'Generating TF-IDF matrix...'
vectorizer = TfidfVectorizer(min_df=2, use_idf=True, smooth_idf = True, sublinear_tf=True, ngram_range=(1,2), norm='l2')
train_vectors = vectorizer.fit_transform(xTr)
test_vectors = vectorizer.transform(xVal)

clf = SGDClassifier(penalty="l2",loss="log",fit_intercept=True, shuffle=True,n_iter=20, n_jobs=-1,alpha=0.000005)
print 'Fitting classifier...'
clf.fit(train_vectors, yTr)

print 'Scoring classifier...'
score = clf.score(test_vectors, yVal)       

print 'Model accuracy: %f' % (score)


