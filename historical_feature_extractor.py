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

project_information = dict()

teacher_totals = dict()
teacher_donation_counts = dict()
teacher_exciting_count = dict()

area_totals = dict()
area_exciting_count = dict()
area_donation_counts = dict()

for row in projects_df.itertuples()

for row in donations_df.itertuples():
	if row.is_teacher_acct == 'f':
		continue
	
	id = row.donor_acctid
	#if we haven't seen the id, put it in the dictionary
	if not id in teachers:
		teachers.update({id: 0})

	teachers[id] = teachers[id] + row.donation_total

print(teachers)
#Calculate information about teachers and projects
teachers_and_projects = projects_df[['projectid', 'teacher_acctid']]

#print(teachers_and_projects.groupby('teacher_acctid'))