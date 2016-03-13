import csv
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn import cross_validation
from sklearn.ensemble import GradientBoostingClassifier

pd.options.mode.chained_assignment = None  # default='warn'

# load the data
print('Loading data...')
projects_df = pd.read_csv('projects.csv')
# donations_df = pd.read_csv('donations.csv')

print('Filling missing data...')
# fill the missing data with previous observation data
projects_df = projects_df.fillna(method='pad')


print('Splitting data...')
# split the training data and testing data

#preprocessing the data based on different types of attributes
projects_numeric_columns = ['school_latitude', 'school_longitude',
                            'fulfillment_labor_materials',
                            'total_price_excluding_optional_support',
                            'total_price_including_optional_support']

projects_id_columns = ['teacher_acctid', 'schoolid', 'school_ncesid']

projects_categorial_columns = np.array(list(set(projects_df.columns).difference(set(projects_numeric_columns)).difference(set(projects_id_columns))))
projects_categorial_values = projects_df[projects_categorial_columns]

print('Label encoding...')
# encode each categorical column
# replace each label with its frequency across training and testing data
for key in projects_categorial_columns:
	if key in ['projectid', 'date_posted']: continue

	projects_categorial_values[key] = projects_df.groupby(key)[key].transform('count')

projects_categorial_values.to_csv('categorical_features.csv')