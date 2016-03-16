print('Asking cleverer people for help...')
import csv
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn import cross_validation
from sklearn.ensemble import GradientBoostingClassifier

pd.options.mode.chained_assignment = None  # default='warn'

def add_or_update(dataFrame, row):

	pass

# load the data
print('Loading data...')
projects_df = pd.read_csv('projects_small.csv')

donations_chunker = 
donations_df = pd.read_csv('donations_small.csv')

teachers_df = pd.DataFrame(columns=('teacher_acctid', 'teacher_exciting_projects', 'teacher_total_donations', 'teacher_donations_count', 'teacher_average_donation'))
schools_df = pd.DataFrame(columns=('schoolid', 'school_exciting_projects', 'school_total_donations', 'school_donations_count', 'school_average_donation'))
subjects_df = pd.DataFrame(columns=('primary_focus_subject', 'subject_exciting_projects', 'subject_total_donations', 'subject_donations_count', 'subject_average_donation'))
grades_df = pd.DataFrame(columns=('grade_level', 'grade_exciting_projects', 'grade_total_donations', 'grade_donations_count', 'grade_average_donation'))

#Combine join projects and donations on project id
print('Thinking about what I need...')
projects_donations_df = pd.merge(projects_df, donations_df, on='projectid');
print(projects_donations_df)
print('Calculating clever things...')
for row in projects_donations_df.iterrows():
	amount = row.dollar_amount
	print("Foo")

	#If this donation if from the teacher associated with the project
	if row.teacher_acctid == row.donor_acctid:
		print("Foo")
		pass

#Put all the data together
print('Putting data together...')
result_df = projects_df.merge(teachers_df, on='teacher_acctid')
result_df = result_df.merge(schools_df, on='schoolid')
result_df = result_df.merge(subjects_df, on='primary_focus_subject')
result_df = result_df.merge(grades_df, on='grade_level')
wanted_columns = list(set(result_df.columns).difference(set(projects_df.columns)).union(set(['projectid'])))
result_df = result_df[wanted_columns]

print('Writing historical features to CSV...')
result_df.to_csv('historical_features.csv')
