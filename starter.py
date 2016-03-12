import csv
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn import cross_validation
from sklearn.ensemble import GradientBoostingClassifier

# load the data
print('Loading data...')
projects = pd.read_csv('projects.csv')
outcomes = pd.read_csv('outcomes.csv')
print('Data loaded...')

# sort the data based on id
projects = projects.sort_values(by='projectid')
outcomes = outcomes.sort_values(by='projectid')

print ('Projects shape', projects.shape)
print ('Outcomes shape', outcomes.shape)

# split the training data and testing data
dates = np.array(projects.date_posted)
train_idx = np.where(dates < '2014-01-01')[0]
test_idx = np.where(dates >= '2014-01-01')[0]

# fill the missing data with previous observation data
projects = projects.fillna(method='pad')

#preprocessing the data based on different types of attributes
projects_numeric_columns = ['school_latitude', 'school_longitude',
                            'fulfillment_labor_materials',
                            'total_price_excluding_optional_support',
                            'total_price_including_optional_support']

projects_id_columns = ['projectid', 'teacher_acctid', 'schoolid', 'school_ncesid']

projects_categorial_columns = np.array(list(set(projects.columns).difference(set(projects_numeric_columns)).difference(set(projects_id_columns)).difference(set(['date_posted']))))
projects_categorial_values = np.array(projects[projects_categorial_columns])

print('Label encoding...')

# encode each categorical column and store in projects_data
label_encoder = LabelEncoder()
projects_data = label_encoder.fit_transform(projects_categorial_values[:,0])

for i in range(1, projects_categorial_values.shape[1]):
    label_encoder = LabelEncoder()
    projects_data = np.column_stack((projects_data, label_encoder.fit_transform(projects_categorial_values[:,i])))

projects_data = projects_data.astype(float)
print('projects_data shape after label encoding', projects_data.shape)

#Predicting
xTr = projects_data[train_idx]
yTr = np.array(outcomes.is_exciting)
xTe = projects_data[test_idx]

print('Splitting training set into training and validation sets...')
xTrain, xVal, yTrain, yVal = cross_validation.train_test_split(xTr, yTr, test_size=0.2, random_state=42)

print('Training gradient boosting classifier...')
clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=3, random_state=0)
clf.fit(xTrain, yTrain)
score = clf.score(xVal, yVal)       

print ('Model accuracy', score)



