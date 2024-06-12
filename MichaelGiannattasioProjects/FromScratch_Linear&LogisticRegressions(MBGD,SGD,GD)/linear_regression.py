# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 14:10:04 2023

@author: 2mgia
"""

#import packages
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

#define the loss
def loss(X, y, theta):
    loss = 0
    for xi, yi in zip(X,y):
        loss += (yi - np.inner(xi, theta))**2
    return loss/X.shape[0] #returns loss

#define gradient
def grad(X, y, theta):
    grad = 0 
    for xi, yi in zip(X,y):
        grad += xi * (yi - np.inner(xi, theta))
    return -2*grad/X.shape[0] #returns gradient

#load in data from filepath in directory
filepath = 'C:/Users/2mgia/Dropbox/AirQualityUCInew.csv'
dataDf = pd.read_csv(filepath) #reads in data

#cleans data structure/formatting
dataDf.replace(to_replace = 200, value = np.nan, inplace = True)
dataDf.drop(labels=["Date","Time","Unnamed: 15", "Unnamed: 16"], axis = 1, inplace = True)
dataDf.dropna(axis=0, inplace=True)

#seperates data into features and target
yKey = "C6H6(GT)" #target variable col-name
y = dataDf[yKey].values #gets target var in y
dataKeys = list(dataDf.keys())
dataKeys.remove(yKey)
X=dataDf[dataKeys].values #gets feature variables in X


#initialize plot data dictionaries for plot data
lossDict = {}
timeDict= {}

#algorithm parameters
max_iter = 1000 
batch_size = 50
alpha = 1e-09
N, d = X.shape
algos = ['GD', 'SGD', 'MBGD']


#initialize the dictionaries with proper lengths
for algo in algos:
    lossDict[algo] = np.zeros(max_iter+1)
    timeDict[algo] = np.zeros(max_iter+1)

#linear regression implementation (uses if statement to do different algorithms)
for algo in algos:
    theta = np.zeros(d) #first guess of theta
    t=time.time() #get time for cpu time plot
    lossDict[algo][0] = loss(X, y, theta) #update initial loss function value
    for k in range(1, max_iter+1): #loops through k iterations
        if algo == "GD": #gradient descent
            gradientIter = grad(X, y, theta) #updates gradient
        elif algo == "SGD":
            i = np.random.choice(N) #gets random value
            gradientIter=grad(np.array([X[i]]), np.array([y[i]]), theta) #updates gradient on RV
        elif algo == "MBGD":
            batch = np.random.choice(N, size = batch_size) #selects batch of size batch_size randomly
            gradientIter = grad(X[batch], y[batch], theta) #updates gradient on batch
        #descent step
        theta -= alpha * gradientIter #recomputes theta
        timeDict[algo][k] = time.time()-t #time stored
        lossDict[algo][k] = loss(X, y, theta) #loss stored

#plot iteration index vs. objective error 
plt.style.use('ggplot')
plt.rc('axes', labelsize = 20)
plt.rc('xtick', labelsize = 20)
plt.rc('ytick', labelsize = 20)
plt.rc('legend', fontsize = 18)
plt.rc('lines', linewidth = 4)
fig = plt.figure(figsize=(8,6))

for algo in algos:
    if algo == 'MBGD':
        plt.plot(lossDict[algo], "--", label = f"{algo} (batch size = {batch_size})")
    else:
        plt.plot(lossDict[algo], label = f"{algo}")
        
plt.legend()
plt.xlabel("Iteration Index", fontsize=14)
plt.ylabel("Objective Error", fontsize=14)
plt.title("Iteration Index vs. Objective Error", fontsize=24)
plt.show()



#plot CPU Time vs. Objective Error
plt.style.use('ggplot')
plt.rc('axes', labelsize = 20)
plt.rc('xtick', labelsize = 20)
plt.rc('ytick', labelsize = 20)
plt.rc('legend', fontsize = 18)
plt.rc('lines', linewidth = 4)
fig = plt.figure(figsize=(8,6))

for algo in algos:
    if algo == 'MBGD':
        plt.plot(timeDict[algo], lossDict[algo], "--", label = f"{algo} (batch size = {batch_size})")
    else:
        plt.plot(timeDict[algo], lossDict[algo],  label = f"{algo}")
        
plt.legend()
plt.xlabel("CPU Time", fontsize=14)
plt.ylabel("Objective Error", fontsize=14)
plt.title("CPU Time vs. Objective Error", fontsize=24)
plt.show()