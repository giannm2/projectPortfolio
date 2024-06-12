# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 21:34:48 2023

@author: 2mgia
"""

import numpy as np
from scipy.stats import norm
# import matplotlib.pyplot as plt

class GaussianNaiveBayes:
    #define function to compute all necessary values to fit to multivariate gaussian distr. (mean, sd) and prior probability for NB
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.mean = {}
        self.std = {}
        self.priors = {}
        
        for cls in self.classes:
            combined = np.hstack((X,y)) #necessary to allow for filtering of X on y
            X_cls = combined[combined[:,57] == cls] #does filtering to get descriptors of data in each class
            X_cls = np.delete(X_cls,-1) #returns X to normal
            self.mean[cls] = np.mean(X_cls, axis=0) #gets conditional mean
            self.std[cls] = np.std(X_cls, axis=0) #gets conditional SD
            self.priors[cls] = len(X_cls) / len(X) #gets conditional percentage
    
    #actual workhorse of naive bayes after computation steps done
    def predict(self, X):
        predictions = []
        for x in X: #loops through test data individually
            posteriors = []
            for cls in self.classes: #calculates probability for each class (1 and 0)
                likelihood = np.sum(np.log(norm.pdf(x, loc=self.mean[cls], scale=self.std[cls]))) #fits to normal (multivariate gaussian), then computes likelihood
                posterior = np.log(self.priors[cls]) + likelihood #uses log-form eq. for posterior probability
                posteriors.append(posterior) #adds posterior prob
            predictions.append(self.classes[np.argmax(posteriors)]) #prediction is the argmax probaility between classes
        return predictions

#multivariate gaussian naive bayes function (main focus)
def naiveBayesGaussian(filename, num_splits, train_percentages):
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
    
    #start naive bayes
    errors = []
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
            
            #Train Naive Bayes
            gnb = GaussianNaiveBayes()
            gnb.fit(X_train, y_train)
            
            #Predict w/ Naive Bayes
            predictions = gnb.predict(X_test)
            error = np.mean(predictions != y_test) #accuracy
            error_per_split.append(error)
        
        mean_error = np.mean(error_per_split) #mean error
        std_dev = np.std(error_per_split) #sd error
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

    # plt.errorbar(x = train_percentages, y = means, yerr = stdevs, capsize = 3)
            
    # plt.legend()
    # plt.ylim(0.40,0.45)
    # plt.xlabel("Training Percent", fontsize=14)
    # plt.ylabel("Mean Prediction Accuracy", fontsize=14)
    # plt.title("Training Percent vs. Mean Prediction Accuracy - Gaussian Naive Bayes", fontsize=24)
    # plt.show()
    
    return #end

# naiveBayesGaussian("spambase.data", 100, [5, 10, 15, 20, 25, 30])