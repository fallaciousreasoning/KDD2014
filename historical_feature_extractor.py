#TODO
"""
	exciting projects/area
	exciting projects/teacher
"""

import csv
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn import cross_validation
from sklearn.ensemble import GradientBoostingClassifier

pd.options.mode.chained_assignment = None  # default='warn'

# load the data
print('Loading data...')

projects_df = pd.read_csv('projects_small.csv')
donations_df = pd.read_csv('donations_small.csv')

#Calculate information about teachers and projects
teachers_and_projects = projects_df[['projectid', 'teacher_acctid']]

print(teachers_and_projects.groupby('teacher_acctid'))