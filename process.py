import pandas as pd
import numpy as np
from sklearn import svm
from sklearn import preprocessing
from sklearn import cross_validation

# load the data and process
print('Loading the data...')
outcomes = pd.read_csv('outcomes.csv')
outcomes = outcomes.sort_values(by='projectid')
outcomes = outcomes.dropna()
outcomes = outcomes.replace(["f", "t"],[0,1])

print('Complete..')

x = outcomes[['at_least_1_teacher_referred_donor', 'fully_funded', 'at_least_1_green_donation', 'great_chat', 'three_or_more_non_teacher_referred_donors', 'one_non_teacher_referred_donor_giving_100_plus', 'donation_from_thoughtful_donor']]
y = outcomes['is_exciting']

x = x.as_matrix()
x = preprocessing.scale(x)
y = y.as_matrix()

xTr, xTe, yTr, yTe = cross_validation.train_test_split(x, y, test_size=0.4, random_state=42)

print xTr.shape
print yTr.shape
print xTe.shape
print yTe.shape

model = svm.SVC(kernel='linear', C=0.01)
model.fit(xTr, yTr)

print model.score(xTe, yTe)

# model2 = svm.SVC(kernel='linear', C=1)
# scores = cross_validation.cross_val_score(model2, x, y, cv=5)
# print scores


