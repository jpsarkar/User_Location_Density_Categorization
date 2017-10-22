#!/usr/bin/env python
#   Author: Jnanendra Sarkar
#
################################################################

import pandas as pd
import numpy as np
import csv
import random
import math
import haversine as hs
import pyspark.sql
import scipy.spatial.distance as sp
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
sc = SparkContext('local')
spark = SparkSession(sc)

iFile = 'sample.csv'
oFile ='sample_output.csv'
MAX_ITR = 100

def minDistMatrix(DS, center, k):
    D = sp.cdist(DS, center, 'euclidean')    
    label = D.argmin(axis=1)
    Dro, Dco = D.shape
    minDist = np.array([ [0] * Dco ] * Dro)
    j = 0
    while (j < k):
        TempList = np.where(label == j)
        for p in TempList[0]:
            minDist[p,j] = D[p,j]
        j += 1
    return minDist
    
def getFitness(mDist):
    return sum(sum(mDist))


    
def chkTermination(oldCenter, currCenter, itr):
    if itr > MAX_ITR: return True
    return oldCenter.tolist() == currCenter.tolist()

def getClusterLabel(DS, center):
    #print("Compute distance matrix...")
    D = sp.cdist(DS, center, 'euclidean')    
    label = D.argmin(axis=1)
    return label

def updateCenter(DS, center, label, k):
    i = 0    
    while (i < k):
        TI = np.where(label == i)
        #print("length...")
        #print(len(TI))
        if (i == 0):
            if (len(TI) == 0):
                TC = center[i]
            else:
                TC = np.mean(DS[TI],0)
        else:
            if (len(TI) == 0):
                TC = center[i]
            else:               
                TC = np.vstack((TC,np.mean(DS[TI],0)))
        i += 1
        #print(TC)
    return TC
        
    

df = spark.read.csv(iFile, header=None, inferSchema=True)
data = df.filter((df._c3 != 0) & (df._c4 != 0)).select(df._c3,df._c4)
dataArr = np.array(data.collect())

k = 3
itr = 0
nRec, nDim = dataArr.shape
CArr = dataArr[random.sample(list(range(1, nRec)),k)]
print("Random initialized Centers...")
print(CArr)
ro, co = CArr.shape
oldCArr = np.array([ [0] * co ] * ro)

while not chkTermination(oldCArr, CArr, itr):
    oldCArr = CArr
    itr += 1
    label = getClusterLabel(dataArr, CArr)

    CArr = updateCenter(dataArr, CArr, label, k)    
    fitness = getFitness(minDistMatrix(dataArr, CArr, k))
    print("Iteration : ",itr," - Objective function value : ",fitness)
    
print("Converged Centers...")
print(CArr)
finalData = np.column_stack((dataArr,label))
np.savetxt(oFile, finalData, delimiter=",",fmt="%14.10f,%14.10f,%d")
