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


def read_file():
    data = pd.read_csv("data/train/train.csv")
    print data.describe()

    #y_boxplot(data)
    y_id_scatterplot(data)


    return None,None

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

def main():

    start_time = time.time()
    X,y = read_file()


    print "--- %s Minutes ---" % ((time.time() - start_time)/60)

if __name__ == "__main__": main()