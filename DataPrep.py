#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 23:41:43 2017

@author: Shiradvd
"""


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pylab as P
from sklearn.preprocessing import MinMaxScaler
#f_classif feature selection
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectPercentile, f_classif, mutual_info_classif
#Tree-based feature selection
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
#wrapper approach
#from sklearn.neighbors import KNeighborsClassifier
#from mlxtend.feature_selection import SequentialFeatureSelector as SFS

from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.datasets import make_classification


def typeModification(_df):
    # Identify which of the orginal features are objects
    ObjFeat=df.keys()[_df.dtypes.map(lambda x: x=='object')]
    
    # Transform the original features to categorical &
    # Creat new 'int' features, resp.
    for f in ObjFeat:
        _df[f] = _df[f].astype("category")
        _df[f+"_Int"] = _df[f].cat.rename_categories(range(_df[f].nunique())).astype(int)
        _df.loc[_df[f].isnull(), f+"_Int"] = np.nan #fix NaN conversion

    if 'Vote' in _df.columns:
       y = _df.Vote.values
       
    # Remove category fields
    _df.dtypes[ObjFeat]
    _df = _df.drop(ObjFeat, axis=1)
    return _df,y


def valueModification(_df):
    #TODO: check if we find non-manual implementation
    #fix field 1 - Avg_monthly_expense_when_under_age_21
    #train.Avg_monthly_expense_when_under_age_21.dropna().hist(bins=100)
    #P.show()
    #set wrong values to NAN    
    train[train.Avg_monthly_expense_when_under_age_21<0]= np.nan
    
    #fix field 2 - Avg_Satisfaction_with_previous_vote
    #train.Avg_Satisfaction_with_previous_vote.dropna().hist()
    #P.show()
    #set wrong values to NAN
    train[train.Avg_Satisfaction_with_previous_vote<0]= np.nan

    return _df


def outlierDetection(_df):
    #TODO!!!
    """
    n_neighbors = 5
    
    for i, weights in enumerate(['uniform', 'distance']):
        knn = neighbors.KNeighborsRegressor(n_neighbors, weights=weights)
        y_ = knn.fit(X, y).predict(T)
    
        plt.subplot(2, 1, i + 1)
        plt.scatter(X, y, c='k', label='data')
        plt.plot(T, y_, c='g', label='prediction')
        plt.axis('tight')
        plt.legend()
        plt.title("KNeighborsRegressor (k = %i, weights = '%s')" % (n_neighbors,
                                                                weights))

        plt.show()
    """
    return _df

def fillingUpMissingValues(_df):
    _df_NoNulls = _df.fillna(_df.mean(), inplace=False)
    
    return _df_NoNulls


def normalizeData(_df_NoNulls):
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(_df_NoNulls), columns=_df_NoNulls.columns)
    return df_scaled

#region feature Selection By Variable Ranking

def test_f_scores(_df_X_NoNulls,y):
    # Univariate feature selection with F-test for feature scoring
    # We use the default selection function: the 15% most significant features
    selector = SelectPercentile(f_classif, percentile=15)
    selector.fit(_df_X_NoNulls,y)
    f_scores = selector.scores_
    f_scores /= f_scores.max()
    
    X_indices = np.arange(_df_X_NoNulls.shape[-1])
    
    plt.bar(X_indices , f_scores, width=.3,
            label='f_classif', color='g')
    plt.title("Comparing Univariate Feature Selection")
    plt.xlabel('Feature number')
    plt.yticks(())
    plt.axis('tight')
    plt.legend(loc='upper right')
    plt.show()
    return f_scores,X_indices

def test_MI_scores(_df_X_NoNulls,y,X_indices,f_scores):
    # Univariate feature selection with mutual information for feature scoring
    selector = SelectPercentile(mutual_info_classif, percentile=15)
    selector.fit(_df_X_NoNulls,y)
    MI_scores = selector.scores_
    MI_scores /= MI_scores.max()
    plt.bar(X_indices - .65, f_scores, width=.3,
            label='f_classif', color='g')
    plt.bar(X_indices - .35, MI_scores, width=.3,
            label='MI', color='b')
    plt.title("Comparing Univariate Feature Selection")
    plt.xlabel('Feature number')
    plt.yticks(())
    plt.axis('tight')
    plt.legend(loc='upper right')
    plt.show()

    return MI_scores

def test_tree_weights(_df_X_NoNulls,y,X_indices,f_scores,MI_scores):
    clf = ExtraTreesClassifier()
    clf = clf.fit(_df_X_NoNulls, y)
    tree_weights = clf.feature_importances_  
    tree_weights /= tree_weights.max()
    plt.bar(X_indices - .05, tree_weights, width=.3, 
            label='Tree weight', color='r')
    plt.bar(X_indices - .35, MI_scores, width=.3,
            label='MI', color='b')
    plt.bar(X_indices - .65, f_scores, width=.3,
            label='f_classif', color='g')
    
    plt.title("Comparing Univariate and Embedded Feature Selection")
    plt.xlabel('Feature number')
    plt.yticks(())
    plt.axis('tight')
    plt.legend(loc='upper right')
    plt.show()

    return tree_weights

    
def featureSelectionByVariableRanking(_df_NoNulls, y):
    print('featureSelectionByVariableRanking')
    feat_names = _df_NoNulls.columns.values    
    
    
    #remove Vote if needed
    feat_names = _df_NoNulls.drop(['Vote_Int'], axis=1).columns.values
    _df_X_NoNulls = _df_NoNulls.drop(['Vote_Int'], axis=1).values
    
    
    rank_f_scores,X_indices = test_f_scores(_df_X_NoNulls,y)
    rank_MI_scores = test_MI_scores(_df_X_NoNulls,y,X_indices,rank_f_scores)
    rank_tree_weights = test_tree_weights(_df_X_NoNulls,y,X_indices,rank_f_scores,rank_MI_scores)   
    
    
    
    
    dfRanking = pd.DataFrame({'X_indices':X_indices,'column_name':feat_names,'tree_weights':rank_tree_weights,'MI_scores':rank_MI_scores,'f_scores':rank_f_scores})
    print('featureSelectionByVariableRanking finished')
    return dfRanking
    
#endregion feature Selection By Variable Ranking


def featureSelectionByWrapperMethod(_df_NoNulls, y):
    print('featureSelectionByWrapperMethod')
    
    feat_names = _df_NoNulls.columns.values  
    
    
    #remove Vote if needed
    feat_names = _df_NoNulls.drop(['Vote_Int'], axis=1).columns.values
    _df_X_NoNulls = _df_NoNulls.drop(['Vote_Int'], axis=1).values
    X_indices = np.arange(_df_X_NoNulls.shape[-1])
    
    
    # Create the RFE object and compute a cross-validated score.
    svc = SVC(kernel="linear")
    # The "accuracy" scoring is proportional to the number of correct
    # classifications
    rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(2),
                  scoring='accuracy')
    rfecv.fit(_df_X_NoNulls, y)
    
    print("Optimal number of features : %d" % rfecv.n_features_)
    
    # Plot number of features VS. cross-validation scores
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.show()
    
    
    estimator = SVC(kernel="linear")
    selector = RFECV(estimator, scoring='accuracy')
    selector = selector.fit(_df_X_NoNulls, y)
    
    
    dfRanking = pd.DataFrame({'X_indices':X_indices,'column_name':feat_names,'WrapperRanking':selector.ranking_})
    
    return dfRanking



#main starts here

#loadData
#working_dir = "/Users/Shiradvd/Desktop/ML/Exercise2"
working_dir = "/Users/yanivy/OneDrive - Microsoft/Old Drive/Dropbox/לימודים 2018/AI/Homework/ai_course"
df = pd.read_csv(working_dir+"/ElectionsData.csv", header=0)
train, validate, test = np.split(df.sample(frac=1), [int(.6*len(df)), int(.8*len(df))])

#correct type of each attribute
train,y = typeModification(train)

#data cleansing
train = valueModification(train)
train = outlierDetection(train)


#Imputation
train_NoNulls = fillingUpMissingValues(train)
#print(train_NoNulls.info())


#Normalization (scaling)
train_NoNulls = normalizeData(train_NoNulls)

##Feature Selection

#filter method - Variable Ranking

dfRankingByVariable = featureSelectionByVariableRanking(train_NoNulls,y)
dfWrapper = featureSelectionByWrapperMethod(train_NoNulls,y)
print(dfRankingByVariable)
print(dfWrapper)

dfAllScores = pd.merge(dfRankingByVariable,dfWrapper, on='X_indices')




