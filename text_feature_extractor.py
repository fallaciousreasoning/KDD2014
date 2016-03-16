import re
import csv
import pandas as pd
import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.linear_model import SGDClassifier
# from sklearn import cross_validation
# from sklearn.ensemble import GradientBoostingClassifier


def clean(string):
  string = re.sub(r"\\t", " ", string)   
  string = re.sub(r"\\n", " ", string)   
  string = re.sub(r"\\r", " ", string)   
  string = re.sub(r"[^A-Za-z0-9\']", " ", string)   
  string = re.sub(r"\s{2,}", " ", string)
  return string.strip()

if __name__=="__main__":   
  # load the data
  print 'Loading projects.csv...'
  projects_df = pd.read_csv('projects.csv')

  print 'Loading essays.csv...'
  essays_df = pd.read_csv('essays.csv')
  essays_df = essays_df.fillna("")

  print 'Loading outcomes.csv...'
  outcomes_df = pd.read_csv('outcomes.csv')

  print 'Merging csv files...'
  # df is a left outer join of projects and outcomes on projectid
  df = pd.merge(projects_df, essays_df, how='left', on='projectid')
  df = pd.merge(df, outcomes_df, how='left', on='projectid')

  print('Splitting data...')
  # split the training data and testing data
  dates = np.array(df.date_posted)
  test_idx = np.where(dates >= '2014-01-01')[0]
  val_idx = np.where(dates < '2014-01-01')[0]
  train_idx = np.where(dates < '2013-01-01')[0]
  val_idx = np.setdiff1d(val_idx, train_idx)

  print 'Cleaning text and counting words...'
  data_cols=["title", "short_description", "need_statement", "essay"]
  for col in data_cols:
    df[col] = df[col].apply(clean)
    df[col+"_wc"] = df[col].apply(lambda x: len(x.split()))
    print 'Cleaned and counted %s' % (col)    

  output_cols = ['projectid', 'title_wc', 'short_description_wc', 'need_statement_wc', 'essay_wc']
  data_cols=['title_wc', 'short_description_wc', 'need_statement_wc', 'essay_wc']
  
  df_out = df[output_cols]

  print 'Writing text_features.csv...'
  df_out.to_csv('text_features.csv')

  # X = np.array(df[data_cols])
  # y = np.array(df.is_exciting)

  # xTr = X[train_idx]
  # yTr = y[train_idx]
  # xVal = X[val_idx]
  # yVal = y[val_idx]
  # xTe = X[test_idx]

  # print('Training gradient boosting classifier...')
  # # predicting
  # # use learning_rate = 0.01, max_depth = 7
  # clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=7)
  # clf.fit(xTr, yTr)
  # score = clf.score(xVal, yVal)       

  # print 'Model accuracy: %f' % (score*100)


