import csv
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn import cross_validation
from sklearn.ensemble import GradientBoostingClassifier

pd.options.mode.chained_assignment = None  # default='warn'

# load the data
print('Loading data...')
outcomes_df = pd.read_csv('outcomes.csv')
projects_df = pd.read_csv('projects.csv')
# donations_df = pd.read_csv('donations.csv')

print('Filling missing data...')
# fill the missing data with previous observation data
projects_df = projects_df.fillna(method='pad')


print('Splitting data...')
# split the training data and testing data

dates = np.array(projects_df.date_posted)
out_idx = np.where(dates < '2010-04-01')[0]
train_idx = np.where(dates < '2014-01-01')[0]

# disregard data before April 2010
# train_idx = np.setdiff1d(train_idx, out_idx)

test_idx = np.where(dates >= '2014-01-01')[0]

# df is a left outer join of projects and outcomes on projectid
df = pd.merge(projects_df, outcomes_df, how='left', on='projectid')

#preprocessing the data based on different types of attributes
projects_numeric_columns = ['school_latitude', 'school_longitude',
                            'fulfillment_labor_materials',
                            'total_price_excluding_optional_support',
                            'total_price_including_optional_support']

projects_id_columns = ['projectid', 'teacher_acctid', 'schoolid', 'school_ncesid']

projects_categorial_columns = np.array(list(set(projects_df.columns).difference(set(projects_numeric_columns)).difference(set(projects_id_columns)).difference(set(['date_posted']))))

projects_categorial_values = df[projects_categorial_columns]

print('Label encoding...')
# encode each categorical column
# replace each label with its frequency across training and testing data
for key in projects_categorial_columns:
    projects_categorial_values[key] = df.groupby(key)[key].transform('count')

projects_data = np.array(projects_categorial_values)

#Predicting
xTr = projects_data[train_idx]
yTr = np.array(df.is_exciting[train_idx])
xTe = projects_data[test_idx]

print('Splitting training set into training and validation sets...')
xTrain, xVal, yTrain, yVal = cross_validation.train_test_split(xTr, yTr, test_size=0.3, random_state=42)

print('Training gradient boosting classifier...')
clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.01, max_depth=7, max_features=11)
clf.fit(xTrain, yTrain)
score = clf.score(xVal, yVal)       

print ('Model accuracy', score)



