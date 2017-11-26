print("hello world")



import csv

import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
import pandas as pd


#set your working_dir
working_dir = "/Users/Shiradvd/Desktop/ML/Exercise2"

# For .read_csv, always use header=0 when you know row 0 is the header row
ElectionsData = pd.read_csv(working_dir+"/ElectionsData.csv", header=0)

## First look at the data
ElectionsData.head()

#Training set feature types
ElectionsData.dtypes

#Concise summary of training data
ElectionsData.info()

#Summary statistics of training data, excluding NaN values
ElectionsData.describe()
#caution: there are a lot of missing values in Age, for example (see info()) Since the nulls were left out of the calculation, the summary statistics values e.g. the mean) might be miskeading

#number of distinct Vote values
VoteNum = ElectionsData.Vote.unique().size
uniqueVotes = ElectionsData.Vote.unique()
print(VoteNum) #10

#Need to update ??? to the party name
for i in range(0,VoteNum):
    print uniqueVotes[i], len(ElectionsData[ (ElectionsData['Gender'] == 'Male') & (uniqueVotes[i] == ElectionsData['Vote'])]),len(ElectionsData[ (ElectionsData['Gender'] == 'Female') & (uniqueVotes[i] == ElectionsData['Vote'])])
#looks like male & female are same    
    
    
## Preparing the data for Machine Learning
### Cleaning the data, creating new features, transforming to numeric values, droping NaNs

#copyDate before any change is done
df3 = ElectionsData.copy(deep = True)
dfElections.head()
#Adding in new 'Gender' column to the dataframe
dfElections['Gender_int'] = dfElections['Sex'].map( {'female':0, 'male':1}).astype(int)

#Let's creat a crosstabcross-tabulation to look at this transformation
pd.crosstab(train.Gender, train.Sex, rownames=['Gender'], colnames=['Sex'])    

#returns count of all the missing values
df2.isnull().sum()

#returns all the rows where Occupation_Satisfaction is missing
df2[df2.Occupation_Satisfaction.isnull()]

#returns how many values from each value in the column Vote
df2['Vote'].value_counts(dropna=False)

#fill NA values
df2[Occupation_Satisfaction].fillna(value='', inplace=True)

#fill with previous row value
df2[Occupation_Satisfaction].fillna(method="ffill")


#fill with next row value
df2[Occupation_Satisfaction].fillna(method="bfill")

#filling all missing values with avarage of before and after values
new_df = df.interpolate()


-------------------------------
df3 = ElectionsData.copy(deep = True)
df3.isnull().sum()
df3.info()
df3.describe(include='all')


# Identify which of the orginal features are objects
ObjFeat=df3.keys()[df3.dtypes.map(lambda x: x=='object')]




# Transform the original features to categorical
# Creat new 'int' features, resp.
for f in ObjFeat:
    df3[f] = df3[f].astype("category")
    df3[f+"_Int"] = df3[f].cat.rename_categories(range(df3[f].nunique())).astype(int)
    df3.loc[df3[f].isnull(), f+"_Int"] = np.nan #fix NaN conversion


print(ObjFeat)
df3.dtypes
df3.head()

# Remove category fields
df3.dtypes[ObjFeat]
df3 = df3.drop(ObjFeat, axis=1)
df3.info()


#fill missing values with mean
df3_NoNulls = df3.fillna(df3.mean(), inplace=False)
df3_NoNulls.describe()


==========================================

# charts
import pylab as P
df3_NoNulls.Occupation_Satisfaction.dropna().hist(bins=10, range= (0,5), alpha=0.5)
P.show()


df3.Avg_Satisfaction_with_previous_vote.dropna().hist(bins=100)
P.show()

df3[df3.Avg_monthly_expense_when_under_age_21<0]= np.nan


median_ages = np.zeros((2,3))











        

