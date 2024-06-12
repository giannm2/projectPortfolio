# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 18:26:47 2023

@author: 2mgia
"""
#import packages
import numpy as np
# import matplotlib.pyplot as plt

#define gradient
def grad(X, y, theta, lam):
    return np.mean((-y * X * (np.exp(-y.T * np.matmul(X, theta)).T))/(1+(np.exp(-y.T * np.matmul(X, theta)).T)), axis = 0) + lam * theta

#have sigmoid function for easy prediction
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#logistic regression function (main focus)
def logisticRegression(filename, num_splits, train_percentages):
    #load in data from filepath in directory
    with open(filename.strip(), 'r') as file:
        lines = file.read().splitlines()
    
    #split data into y (target) and X (feature variables)
    num_data = len(lines)
    y = [lines[i].split(",")[-1] for i in range(num_data)]
    y = [np.asarray(y[i], dtype = np.float32) for i in range(len(y))]
    y = np.asarray(y)[:,np.newaxis]
    X = [lines[i].split(",")[:-1] for i in range(num_data)]
    X = [np.asarray(X[i], dtype = np.float32) for i in range(len(X))]
    X = np.asarray(X)
    
    
    #algorithm parameters
    max_iter = 10000
    batch_size = 50
    alpha = 1e-9
    lam = 0
    errors = []
    
    #start logistic regression
    for train_percent in train_percentages: #loops through train sizes
        error_per_split = []
        for num in range(num_splits): #do the number of times required
            # Split data into train and test sets
            indices = np.random.permutation(len(X))
            train_size = int(0.8 * len(X)) #80% train
            actual_train_size = int(train_percent/100 * len(X)) #use input to get the necessary percentage of training data used
            train_idx, test_idx = indices[:actual_train_size], indices[train_size:]
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            N, dim = np.shape(X_train) #get dimension of data
            theta = np.zeros(dim) #first guess of theta
            
            for k in range(1, max_iter+1): #loops through k iterations
                batch = np.random.choice(N, size = batch_size) #selects batch of size batch_size randomly
                gradientIter = grad(X_train[batch], y_train[batch], theta, lam) #updates gradient on batch
                #descent step
                theta -= alpha * gradientIter #recomputes theta
            
            #Predict w/ Logistic Regression
            predictions = (sigmoid(np.dot(X_test, theta)) >= 0.5) #uses 50% probability threshold for classification
            error = 1-np.mean(y_test != predictions) #flips probability threshold to accurately predict, then gets percentage
            error_per_split.append(error)
        
        mean_error = np.mean(error_per_split) #mean error
        std_dev = np.std(error_per_split) #st dev error
        errors.append((mean_error, std_dev))
        print(f"Training Set Percentage: {train_percent}%, Mean Error: {mean_error:.4f}, Standard Deviation: {std_dev:.4f}") #output for command line
    
    
    # #plot Training Percent vs. Mean Prediction Accuracy
    # means = [i[0] for i in errors]
    # stdevs = [i[1] for i in errors]
    
    # plt.style.use('ggplot')
    # plt.rc('axes', labelsize = 20)
    # plt.rc('xtick', labelsize = 20)
    # plt.rc('ytick', labelsize = 20)
    # plt.rc('legend', fontsize = 18)
    # plt.rc('lines', linewidth = 4)

    # plt.errorbar(x = train_percentages, y = means, yerr = stdevs)
            
    # plt.legend()
    # plt.ylim(0.37,0.43)
    # plt.xlabel("Training Percent", fontsize=14)
    # plt.ylabel("Mean Prediction Accuracy", fontsize=14)
    # plt.title("Training Percent vs. Mean Prediction Accuracy - Logistic Regression", fontsize=24)
    # plt.show()
    
    return  #end

# logisticRegression("spambase.data", 100, [5, 10, 15, 20, 25, 30])