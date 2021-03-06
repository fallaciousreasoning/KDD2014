#import the things we need
import csv
import pandas as pd
import numpy as np

pd.options.mode.chained_assignment = None  # default='warn'

# load the data
print 'Loading projects.csv...'
projects_df = pd.read_csv('projects.csv')

print 'Filling missing data...'
# fill the missing data with previous observation data
projects_df = projects_df.fillna("")
projects_df["month"] = projects_df["date_posted"].apply(lambda x: x.split("-")[1])

#preprocessing the data based on different types of attributes
projects_numeric_columns = ['school_latitude', 
                            'school_longitude',
                            'fulfillment_labor_materials',
                            'total_price_excluding_optional_support',
                            'total_price_including_optional_support']

projects_id_columns = ['teacher_acctid', 'schoolid', 'school_ncesid']

#load the projects cv and remove all the numeric columns
projects_categorial_columns = np.array(list(set(projects_df.columns).difference(set(projects_numeric_columns)).difference(set(projects_id_columns))))

#take all the values from projects that can be categorical features
projects_categorial_values = projects_df[projects_categorial_columns]

print 'Binarizing categorical values...'
# replace each label with its frequency across training and testing data
for key in projects_categorial_columns:
  if key in ['projectid', 'date_posted', 'students_reached']: 
    continue

  #set the value for text base categories to their number of occurrences throughout the dataset
  projects_categorial_values[key] = projects_df.groupby(key)[key].transform('count')
  print 'Binarized %s' % key

print 'Writing categorical_features.csv...'
#save the data to a categorical data csv
projects_categorial_values.to_csv('categorical_features.csv')