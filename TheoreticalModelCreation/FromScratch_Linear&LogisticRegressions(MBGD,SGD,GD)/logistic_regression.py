# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 15:03:21 2023

@author: 2mgia
"""

#import packages
import numpy as np
from numpy import linalg as LA
import time
import matplotlib.pyplot as plt

#define the loss
def loss(X, y, theta, lam):
    return np.mean(np.log(1 + (np.exp(-y.T * np.matmul(X, theta)).T))) + lam/2 * LA.norm(theta)**2

#define gradient
def grad(X, y, theta, lam):
    return np.mean((-y * X * (np.exp(-y.T * np.matmul(X, theta)).T))/(1+(np.exp(-y.T * np.matmul(X, theta)).T)), axis = 0) + lam * theta


#load in data from filepath in directory
with open('C:/Users/2mgia/Dropbox/ionosphere.data', 'r') as file:
    lines = file.read().splitlines()
    
#partition data into features (X) and target (y)
num_data = len(lines)
y = [lines[i].split(",")[-1] for i in range(num_data)]
y = [1 if y == "g" else -1 for i in range(num_data)]
y = np.asarray(y)[:,np.newaxis]
X = [lines[i].split(",")[:-1] for i in range(num_data)]
X = [np.asarray(X[i], dtype = np.float32) for i in range(len(X))]
X = np.asarray(X)


#initialize plot data dictionaries for plot data
lossDict = {}
timeDict= {}

#algorithm parameters
max_iter = 10000
batch_size = 50
alpha = 1e-2
lam = 0
N, dim = np.shape(X)
algos = ['GD', 'SGD', 'MBGD']




#we first use lambda = 0

#initialize the dictionaries with proper lengths
for algo in algos:
    lossDict[algo] = np.zeros(max_iter+1)
    timeDict[algo] = np.zeros(max_iter+1)

#linear regression implementation (uses if statement to do different algorithms)
for algo in algos:
    theta = np.zeros(dim) #first guess of theta
    t=time.time() #get time for cpu time plot
    lossDict[algo][0] = loss(X, y, theta, lam) #update initial loss function value
    for k in range(1, max_iter+1): #loops through k iterations
        if algo == "GD": #gradient descent
            gradientIter = grad(X, y, theta, lam) #updates gradient
        elif algo == "SGD":
            i = np.random.choice(N) #gets random value
            gradientIter=grad(np.array([X[i]]), np.array([y[i]]), theta, lam) #updates gradient on RV
        elif algo == "MBGD":
            batch = np.random.choice(N, size = batch_size) #selects batch of size batch_size randomly
            gradientIter = grad(X[batch], y[batch], theta, lam) #updates gradient on batch
        #descent step
        theta -= alpha * gradientIter #recomputes theta
        timeDict[algo][k] = time.time()-t #time stored
        lossDict[algo][k] = loss(X, y, theta, lam) #loss stored

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
plt.title("Iteration Index vs. Objective Error, lambda = 0", fontsize=24)
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
plt.title("CPU Time vs. Objective Error, lambda = 0", fontsize=24)
plt.show()




#we then use lambda = 0.01
lam = 0.01

#initialize the dictionaries with proper lengths
for algo in algos:
    lossDict[algo] = np.zeros(max_iter+1)
    timeDict[algo] = np.zeros(max_iter+1)

#linear regression implementation (uses if statement to do different algorithms)
for algo in algos:
    theta = np.zeros(dim) #first guess of theta
    t=time.time() #get time for cpu time plot
    lossDict[algo][0] = loss(X, y, theta, lam) #update initial loss function value
    for k in range(1, max_iter+1): #loops through k iterations
        if algo == "GD": #gradient descent
            gradientIter = grad(X, y, theta, lam) #updates gradient
        elif algo == "SGD":
            i = np.random.choice(N) #gets random value
            gradientIter=grad(np.array([X[i]]), np.array([y[i]]), theta, lam) #updates gradient on RV
        elif algo == "MBGD":
            batch = np.random.choice(N, size = batch_size) #selects batch of size batch_size randomly
            gradientIter = grad(X[batch], y[batch], theta, lam) #updates gradient on batch
        #descent step
        theta -= alpha * gradientIter #recomputes theta
        timeDict[algo][k] = time.time()-t #time stored
        lossDict[algo][k] = loss(X, y, theta, lam) #loss stored

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
plt.title("Iteration Index vs. Objective Error, lambda = 0.01", fontsize=24)
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
plt.title("CPU Time vs. Objective Error, lambda = 0.01", fontsize=24)
plt.show()


