import csv
import pandas as pd
import numpy as np
from sklearn import cross_validation
from sklearn.ensemble import GradientBoostingClassifier

# load the data
print('Loading data...')
outcomes_df = pd.read_csv('outcomes.csv')
categorical_df = pd.read_csv('categorical_features.csv')

print('Merging csv files...')
# df is a left outer join of projects and outcomes on projectid
df = pd.merge(categorical_df, outcomes_df, how='left', on='projectid')

print('Splitting data...')
# split the training data and testing data
dates = np.array(df.date_posted)
train_idx = np.where(dates < '2014-01-01')[0]
# disregard data before April 2010
# out_idx = np.where(dates < '2010-04-01')[0]
# train_idx = np.setdiff1d(train_idx, out_idx)
test_idx = np.where(dates >= '2014-01-01')[0]

categorical_cols = np.array(list(set(categorical_df.columns).difference(set(['projectid','date_posted']))))

X = np.array(df[categorical_cols])
y = np.array(df.is_exciting)

#Predicting
xTr = X[train_idx]
yTr = y[train_idx]

xTe = X[test_idx]

print('Splitting training set into training and validation sets...')
xTrain, xVal, yTrain, yVal = cross_validation.train_test_split(xTr, yTr, test_size=0.3, random_state=42)

print('Training gradient boosting classifier...')
# use learning_rate = 0.01, max_depth = 7
clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1, max_depth=3, max_features=11)
clf.fit(xTrain, yTrain)
score = clf.score(xVal, yVal)       

print ('Model accuracy', score)



