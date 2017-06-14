#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: omayr
@description: Exploratody Analysis of the Mercedez-Benz testing system data

"""
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import RandomizedLogisticRegression, LogisticRegression, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import normalize,scale
import xgboost
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier



def read_file():

    data = pd.read_csv("data/train/train.csv")
    # Don't forget to drop columns that are duplicate both
    # in train and test datasets.

    col_to_trans = ['X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X8']

    for col in col_to_trans:
        data[col] = col_transformation(data[col])

    return data.iloc[:,2:378],data['y']

def col_transformation(col):

    le = LabelEncoder()
    le.fit(col)
    col_trans = le.transform(col)

    return col_trans

def visualize_corr(data):

    df = data.drop(['ID', 'y', 'X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X8'],axis=1)
    #print df.corr()
    plt.matshow(df.corr())
    plt.show()
    #sb.heatmap(df.corr())#, xticklabels=df.columns.values, yticklabels=df.columns.values)

def y_id_scatterplot(data):

    id = [x for x in range(0,len(data))]
    y = np.array(data['y'])
    plt.scatter(id,np.sort(y))
    plt.show()

def y_boxplot(data):

    # creates boxplot for the response variable
    y = np.array(data['y'])
    fig = plt.figure(1, figsize=(9, 6))
    ax = fig.add_subplot(111)
    bp = ax.boxplot(y)
    plt.show(bp)

def feature_selection(X,y):

    model = RandomizedLogisticRegression(C=0.1,
                                       sample_fraction=0.75,
                                       n_resampling=10, selection_threshold=0.2, n_jobs=1,
                                       random_state=42, verbose=True)

    print "fitting started"
    model.fit(X,y)
    return model.get_support(indices=True)

def pca(X,y):

    LR = LogisticRegression()
    LR.C = 0.0001
    LR.penalty = 'l2'
    pca = PCA()
    pipe = Pipeline(steps=[('pca', pca), ('LR', LR)])

    n_components = [x for x in range(10, 350, 10)]
    Cs = [10 ** x for x in range(-6, 4, 1)]

    estimator = GridSearchCV(pipe, dict(pca__n_components=n_components, LR__C=Cs), n_jobs=4)
    estimator.fit(X,y)

    return estimator

def basic_regression(X,y):

    models = [xgboost.XGBClassifier( learning_rate =0.1,
                                 n_estimators=140,
                                 max_depth=9,
                                 min_child_weight=1,
                                 gamma=0,
                                 subsample=0.8,
                                 colsample_bytree=0.8,
                                 objective= 'reg:linear',
                                 nthread=4,
                                 scale_pos_weight=1,
                                 seed=27)
    , LinearRegression(), RandomForestRegressor()]
    names = ['XGB','LR','RF']

    for model,name in zip(models,names):
        scores = cross_val_score(model, X, y, cv=4, n_jobs=-1)
        print("Classifier: %s, Acr = %.6f" % (name, np.mean(scores)))



def main():

    start_time = time.time()
    X,y = read_file()

    basic_regression(X,y)
    #feature_index = feature_selection(np.array(X),np.array(y))
    #print feature_index
    #pca(X,y)
    print "--- %s Minutes ---" % ((time.time() - start_time)/60)

if __name__ == "__main__": main()