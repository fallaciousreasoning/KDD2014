print('Asking cleverer people for help...')
import csv
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import GradientBoostingClassifier

# load the data
print('Loading data...')
outcomes_df = pd.read_csv('outcomes.csv')
categorical_df = pd.read_csv('categorical_features.csv')
historical_df = pd.read_csv('historical_features.csv')
essays_df = pd.read_csv('text_features.csv')

sample = pd.read_csv('sampleSubmission.csv')
sample = sample.sort_values(by='projectid')

print('Merging csv files...')
# df is a left outer join of projects and outcomes on projectid
df = pd.merge(categorical_df, outcomes_df, how='left', on='projectid').merge(historical_df, how='left', on='projectid').merge(essays_df, how='left', on='projectid')
df = df.fillna(0)
df = df.sort_values(by='projectid')

print('Splitting data...')
# split the training data and testing data
dates = np.array(df.date_posted)
test_idx = np.where(dates >= '2014-01-01')[0]
train_idx = np.where(dates < '2013-01-01')[0]
val_idx = np.where(dates < '2014-01-01')[0]
val_idx = np.setdiff1d(val_idx, train_idx)

# disregard data before April 2010
out_idx = np.where(dates < '2010-04-01')[0]
train_idx = np.setdiff1d(train_idx, out_idx)

data_cols = ['month',
              'title_wc', 'short_description_wc', 'need_statement_wc', 'essay_wc',
              'subject_total',
              'school_district',
              'subject_average',
              'secondary_focus_area',
              'primary_focus_area',
              'poverty_level',
              'school_year_round',
              'school_charter',
              'school_exciting',
              'teacher_teach_for_america',
              'school_magnet',
              'school_count',
              'grade_level',
              'primary_focus_subject',
              'teacher_average',
              'school_kipp',
              'teacher_total',
              'grade_total',
              'school_total',
              'school_state',
              'grade_average',
              'school_charter_ready_promise',
              'teacher_prefix',
              'school_county',
              'subject_exciting',
              'students_reached',
              'school_city',
              'teacher_exciting',
              'eligible_double_your_impact_match',
              'secondary_focus_subject',
              'teacher_ny_teaching_fellow',
              'subject_count',
              'school_average',
              'grade_exciting',
              'eligible_almost_home_match',
              'school_nlns',
              'school_metro',
              'grade_count',
              'school_zip',
              'teacher_count',
              'resource_type']

X = np.array(df[data_cols])
y = np.array(df.is_exciting)

xTr = X[train_idx]
yTr = y[train_idx]
xVal = X[val_idx]
yVal = y[val_idx]
xTe = X[test_idx]

print 'Selecting features with extra trees and training gradient boosting classifier...'
clf = Pipeline([
  ('feature_selection', SelectFromModel(ExtraTreesClassifier())),
  ('classification', GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=7, max_features=11))
])
clf.fit(xTr, yTr)

score = clf.score(xVal, yVal)


print 'Model accuracy: %f' % (score*100)

preds = clf.predict_proba(xTe)[:,1]

#Save prediction into a file
sample['is_exciting'] = preds

print 'Saving predictions...'
sample.to_csv('gbm_predictions.csv', index = False)

# Score = 0.57613


