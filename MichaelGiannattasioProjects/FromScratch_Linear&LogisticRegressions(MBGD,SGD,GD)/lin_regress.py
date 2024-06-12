# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 14:10:04 2023

@author: 2mgia
"""

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
        grad += (yi - np.inner(xi, theta))
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

#algorithm parameters
max_iter = 1000
batch_size = 50
alpha = 1e-06
N, d = X.shape
algos = ['GD', 'SGD', 'MBGD']


#initialize plot data dictionaries
losses = {}
cputs= {}

#initialize the dictionaries with proper lengths
for algo in algos:
    losses[algo] = np.zeros(max_iter+1)
    cputs[algo] = np.zeros(max_iter+1)

#linear regression implementation (uses if statement to do different algorithms)
for algo in algos:
    theta = np.zeros(d) #first guess of theta
    t=time.time() #get time for cpu time plot
    losses[algo][0] = loss(X, y, theta) #update initial loss function value
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
        cputs[algo][k] = time.time()-t #time stored
        losses[algo][k] = loss(X, y, theta) #loss stored

#plot objective error vs. iteration index
plt.style.use('ggplot')
plt.rc('axes', labelsize = 20)
plt.rc('xtick', labelsize = 20)
plt.rc('ytick', labelsize = 20)
plt.rc('legend', fontsize = 18)
plt.rc('lines', linewidth = 4)
fig = plt.figure(figsize=(8,6))

for algo in algos:
    if algo == 'MBGD':
        plt.plot(losses[algo], "--", label = f"{algo} (batch size = {batch_size})")
    else:
        plt.plot(losses[algo], label = f"{algo}")
        
plt.legend()
plt.xlabel("Iteration Index", fontsize=14)
plt.ylabel("Objective Error", fontsize=14)
plt.title("Objective Error vs. Iteration Index", fontsize=24)
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
        plt.plot(losses[algo], "--", label = f"{algo} (batch size = {batch_size})")
    else:
        plt.plot(losses[algo], label = f"{algo}")
        
plt.legend()
plt.xlabel("Objective Error", fontsize=14)
plt.ylabel("CPU Time", fontsize=14)
plt.title("CPU Time vs. Objective Error", fontsize=24)
plt.show()