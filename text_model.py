import re
import csv
import pandas as pd
import numpy as np
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
  # load the data
  print 'Loading projects.csv...'
  projects_df = pd.read_csv('projects.csv')

  print 'Loading essays.csv...'
  essays_df = pd.read_csv('essays.csv')
  essays_df = essays_df.fillna("")

  print 'Loading outcomes.csv...'
  outcomes_df = pd.read_csv('outcomes.csv')

  sample = pd.read_csv('sampleSubmission.csv')
  sample = sample.sort_values(by='projectid')

  print 'Merging csv files...'
  # df is a left outer join of projects and outcomes on projectid
  df = projects_df.merge(essays_df, how='left', on='projectid').merge(outcomes_df, how='left', on='projectid')
  df = df.sort_values(by='projectid')

  print('Splitting data...')
  # split the training data and testing data
  dates = np.array(df.date_posted)
  test_idx = np.where(dates >= '2014-01-01')[0]
  val_idx = np.where(dates < '2014-01-01')[0]
  train_idx = np.where(dates < '2013-01-01')[0]
  val_idx = np.setdiff1d(val_idx, train_idx)

  out_idx = np.where(dates < '2010-04-01')[0]
  train_idx = np.setdiff1d(train_idx, out_idx)

  print 'Cleaning text'
  data_cols=["title", "short_description", "need_statement", "essay"]
  for col in data_cols:
    df[col] = df[col].apply(clean)
    print 'Cleaned %s' % (col)  

  X = np.array(df.essay)
  y = np.array(df.is_exciting)
  xTr = X[train_idx]
  yTr = y[train_idx]
  xVal = X[val_idx]
  yVal = y[val_idx]

  xTe = X[test_idx]

  print 'Generating TF-IDF matrix...'
  vectorizer = TfidfVectorizer(min_df=2, analyzer='word', stop_words=stopwords, sublinear_tf=True, ngram_range=(1,2), norm='l2', token_pattern=r"(?u)\b[A-Za-z0-9()\'\-?!\"%]+\b")

  xTr = vectorizer.fit_transform(xTr)
  xVal = vectorizer.transform(xVal)
  xTe = vectorizer.transform(xTe)

  clf = SGDClassifier(penalty="l2",loss="log",fit_intercept=True, shuffle=True,n_iter=50, alpha=0.000005)
  print 'Fitting classifier...'
  clf.fit(xTr, yTr)

  print 'Scoring classifier...'
  score = clf.score(xVal, yVal)    

  print 'Model accuracy: %f' % (score*100)

  preds = clf.predict_proba(xTe)[:,1]

  #Save prediction into a file
  sample['is_exciting'] = preds

  print 'Saving predictions...'
  sample.to_csv('text_predictions.csv', index = False)

  # Score = 0.56924 


