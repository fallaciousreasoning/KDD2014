print('Asking cleverer people for help...')
import csv
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn import cross_validation
from sklearn.ensemble import GradientBoostingClassifier

pd.options.mode.chained_assignment = None  # default='warn'

teachers_data = dict()
schools_data = dict()
subjects_data= dict()
grades_data = dict()


def default_key_map(prefix, key):
    return prefix + key


def teacher_key_map(key):
    if key == 'id': return 'teacher_acctid'
    return default_key_map('teacher_', key)


def school_key_map(key):
    if key == 'id': return 'schoolid'
    return default_key_map('school_', key)


def subject_key_map(key):
    if key == 'id': return 'primary_focus_subject'
    return default_key_map('subject_', key)


def grade_key_map(key):
    if key == 'id': return 'grade_level'
    return default_key_map('grade_', key)


def build_entry(id, total, keymap):
    return {keymap('id'):id, keymap('total'):total, keymap('count'):1, keymap('average'):total}


def add_or_update(data, row, keymap):
    id = keymap('id')
    average = keymap('average')
    total = keymap('total')
    count = keymap('count')

    if not row[id] in data:
        data.update({row[id]: row })
        return

    previous_data = data[row[id]]
    previous_data[total] += row[total]
    previous_data[count] += 1
    previous_data[average] = previous_data[total] / previous_data[count]

# load the data
print('Loading data...')
projects_df = pd.read_csv('projects_small.csv')
donations_df = pd.read_csv('donations_small.csv')

# Combine join projects and donations on project id
print('Thinking about what I need...')
projects_donations_df = pd.merge(projects_df, donations_df, on='projectid');

print('Calculating clever things...')
for row in projects_donations_df.itertuples():
    amount = row.donation_total

    # If this donation if from the teacher associated with the project
    if row.teacher_acctid == row.donor_acctid:
        entry = build_entry(row.teacher_acctid, amount, teacher_key_map)
        add_or_update(teachers_data, entry, teacher_key_map)

    school_entry = build_entry(row.schoolid, amount, school_key_map)
    add_or_update(schools_data, school_entry, school_key_map)

    subject_entry = build_entry(row.primary_focus_subject, amount, subject_key_map)
    add_or_update(subjects_data, subject_entry, subject_key_map)

    grade_entry = build_entry(row.grade_level, amount, grade_key_map)
    add_or_update(grades_data, grade_entry, grade_key_map)

teachers_df = pd.DataFrame(list(teachers_data.values()))
schools_df = pd.DataFrame(list(schools_data.values()))
subjects_df = pd.DataFrame(list(subjects_data.values()))
grades_df = pd.DataFrame(list(grades_data.values()))

# Put all the data together
print('Putting data together...')
result_df = projects_df.merge(teachers_df, on='teacher_acctid')
result_df = result_df.merge(schools_df, on='schoolid')
result_df = result_df.merge(subjects_df, on='primary_focus_subject')
result_df = result_df.merge(grades_df, on='grade_level')
wanted_columns = list(set(result_df.columns).difference(set(projects_df.columns)).union(set(['projectid'])))
result_df = result_df[wanted_columns]

print('Writing historical features to CSV...')
result_df.to_csv('historical_features.csv', index=False)
