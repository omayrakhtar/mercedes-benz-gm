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
from sklearn import utils
from sklearn.linear_model import RandomizedLogisticRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.preprocessing import normalize,scale,LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV,StratifiedKFold,cross_val_score,train_test_split
import xgboost


def read_file():
    data = pd.read_csv("data/train/train.csv")
    #print (set(data['y']))
    print len(data)
    print len(list(data))
    print list(data)

    return None,None

def main():

    start_time = time.time()
    X,y = read_file()


    print "--- %s Minutes ---" % ((time.time() - start_time)/60)

if __name__ == "__main__": main()