print 'Asking cleverer people for help...'
import re
import csv
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier

def clean(string):
  string = re.sub(r"\\t", " ", string)   
  string = re.sub(r"\\n", " ", string)   
  string = re.sub(r"\\r", " ", string)   
  string = re.sub(r"[^A-Za-z0-9\']", " ", string)   
  string = re.sub(r"\s{2,}", " ", string)
  return string.strip()

if __name__=="__main__":
  
  data_cols1 = ['month',
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

  stopwords = ['a', 'about', 'above', 'across', 'after', 'afterwards', 'again', 'against', 'all', 'almost', 'alone', 'along', 'already', 'also', 'although', 'always', 'am', 'among', 'amongst', 'amoungst',
                'amount', 'an', 'and', 'another', 'any', 'anyhow', 'anyone', 'anything', 'anyway', 'anywhere', 'are', 'around', 'as', 'at', 'back', 'be', 'became', 'because', 'become', 'becomes', 'becoming',
                'been', 'before', 'beforehand', 'behind', 'being', 'below', 'beside', 'besides', 'between', 'beyond', 'bill', 'both', 'bottom', 'but', 'by', 'call', 'can', 'cannot', 'cant', 'co', 'computer',
                'con', 'could', 'couldnt', 'cry', 'de', 'describe', 'detail', 'do', 'done', 'down', 'due', 'during', 'each', 'eg', 'eight', 'either', 'eleven', 'else', 'elsewhere', 'empty', 'enough', 'etc',
                'even', 'ever', 'every', 'everyone', 'everything', 'everywhere', 'except', 'few', 'fifteen', 'fify', 'fill', 'find', 'fire', 'first', 'five', 'for', 'former', 'formerly', 'forty', 'found', 
                'four', 'from', 'front', 'full', 'further', 'get', 'give', 'go', 'had', 'has', 'hasnt', 'have', 'he', 'hence', 'her', 'here', 'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herse"', 
                'him', 'himse"', 'his', 'how', 'however', 'hundred', 'i', 'ie', 'if', 'in', 'inc', 'indeed', 'interest', 'into', 'is', 'it', 'its', 'itse"', 'keep', 'last', 'latter', 'latterly', 'least', 
                'less', 'ltd', 'made', 'many', 'may', 'me', 'meanwhile', 'might', 'mill', 'mine', 'more', 'moreover', 'most', 'mostly', 'move', 'much', 'must', 'my', 'myse"', 'name', 'namely', 'neither', 
                'never', 'nevertheless', 'next', 'nine', 'no', 'nobody', 'none', 'noone', 'nor', 'not', 'nothing', 'now', 'nowhere', 'of', 'off', 'often', 'on', 'once', 'one', 'only', 'onto', 'or', 'other', 
                'others', 'otherwise', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'part', 'per', 'perhaps', 'please', 'put', 'rather', 're', 'same', 'see', 'seem', 'seemed', 'seeming', 'seems', 
                'serious', 'several', 'she', 'should', 'show', 'side', 'since', 'sincere', 'six', 'sixty', 'so', 'some', 'somehow', 'someone', 'something', 'sometime', 'sometimes', 'somewhere', 'still', 
                'such', 'system', 'take', 'ten', 'than', 'that', 'the', 'their', 'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', 'therefore', 'therein', 'thereupon', 'these', 'they', 
                'thick', 'thin', 'third', 'this', 'those', 'though', 'three', 'through', 'throughout', 'thru', 'thus', 'to', 'together', 'too', 'top', 'toward', 'towards', 'twelve', 'twenty', 'two', 'un', 
                'under', 'until', 'up', 'upon', 'us', 'very', 'via', 'was', 'we', 'well', 'were', 'what', 'whatever', 'when', 'whence', 'whenever', 'where', 'whereafter', 'whereas', 'whereby', 'wherein', 
                'whereupon', 'wherever', 'whether', 'which', 'while', 'whither', 'who', 'whoever', 'whole', 'whom', 'whose', 'why', 'will', 'with', 'within', 'without', 'would', 'yet', 'you', 'your', 'yours',
                'yourself', 'yourselves']

  print 'Loading projects.csv...'
  projects_df = pd.read_csv('projects.csv')

  print 'Loading essays.csv...'
  essays_df = pd.read_csv('essays.csv')
  essays_df = essays_df.fillna("")

  print 'Loading outcomes.csv...'
  outcomes_df = pd.read_csv('outcomes.csv')

  print 'Loading categorical_features.csv...'
  categorical_df = pd.read_csv('categorical_features.csv')

  print 'Loading historical_features.csv...'
  historical_df = pd.read_csv('historical_features.csv')

  print 'Loading text_features.csv...'
  text_df = pd.read_csv('text_features.csv')

  print 'Loading sampleSubmission.csv.csv...'
  sample = pd.read_csv('sampleSubmission.csv')
  sample = sample.sort_values(by='projectid')

  print('Merging csv files...')
  df1 = pd.merge(categorical_df, outcomes_df, how='left', on='projectid').merge(historical_df, how='left', on='projectid').merge(text_df, how='left', on='projectid')
  df1 = df1.fillna(0)
  df1 = df1.sort_values(by='projectid')

  df2 = projects_df.merge(essays_df, how='left', on='projectid').merge(outcomes_df, how='left', on='projectid')
  df2 = df2.sort_values(by='projectid')

  print('Splitting data...')
  dates = np.array(df1.date_posted)
  test_idx = np.where(dates >= '2014-01-01')[0]
  train_idx = np.where(dates < '2013-01-01')[0]
  val_idx = np.where(dates < '2014-01-01')[0]
  val_idx = np.setdiff1d(val_idx, train_idx)

  # disregard data before April 2010
  out_idx = np.where(dates < '2010-04-01')[0]
  train_idx = np.setdiff1d(train_idx, out_idx)

  print 'Cleaning text'
  data_cols2=["title", "short_description", "need_statement", "essay"]
  for col in data_cols2:
    df2[col] = df2[col].apply(clean)
    print 'Cleaned %s' % (col)

  X1 = np.array(df1[data_cols1])
  y1 = np.array(df1.is_exciting)

  xTr1 = X1[train_idx]
  yTr1 = y1[train_idx]
  xTe1 = X1[test_idx]

  X2 = np.array(df2.essay)
  y2 = np.array(df2.is_exciting)

  xTr2 = X2[train_idx]
  yTr2 = y2[train_idx]
  xTe2 = X2[test_idx]

  print 'Selecting features with extra trees and training gradient boosting classifier...'
  clf1 = Pipeline([
    ('feature_selection', SelectFromModel(ExtraTreesClassifier())),
    ('classification', GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=7, max_features=11))
  ])

  clf1.fit(xTr1, yTr1)

  preds1 = clf1.predict_proba(xTe1)[:,1]

  print 'Generating TF-IDF vectors...'
  vectorizer = TfidfVectorizer(min_df=2, analyzer='word', stop_words=stopwords, sublinear_tf=True, ngram_range=(1,2), norm='l2', token_pattern=r"(?u)\b[A-Za-z0-9()\'\-?!\"%]+\b")

  xTr2 = vectorizer.fit_transform(xTr2)
  xTe2 = vectorizer.transform(xTe2)

  print 'Fitting SGD classifier...'
  clf2 = SGDClassifier(penalty="l2",loss="log",fit_intercept=True, shuffle=True,n_iter=50, alpha=0.000005)
  clf2.fit(xTr2, yTr2)

  preds2 = clf2.predict_proba(xTe2)[:,1]

  preds = (2*preds1 + 1*preds2)/3

  sample['is_exciting'] = preds

  sample.to_csv('ensemble_predictions.csv', index = False)

  #score = 0.59085

