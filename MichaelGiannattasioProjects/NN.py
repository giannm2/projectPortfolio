# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 18:54:21 2023

@author: 2mgia
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import math

# Define your neural network models
class Model1(nn.Module):
    def __init__(self):
        super(Model1, self).__init__()
        # Define your model architecture (One-hidden layer NN with ReLU)
        self.fc1 = nn.Linear(in_features=input_size, out_features=10)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features=10, out_features=1)  # Assuming binary classification

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return torch.sigmoid(x)

class Model2(nn.Module):
    def __init__(self):
        super(Model2, self).__init__()
        # Define your model architecture (One-hidden layer NN with Tanh)
        self.fc1 = nn.Linear(in_features=input_size, out_features=10)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(in_features=10, out_features=1)  # Assuming binary classification

    def forward(self, x):
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.fc2(x)
        return torch.sigmoid(x)

class Model3(nn.Module):
    def __init__(self):
        super(Model3, self).__init__()
        # Define your model architecture (One-hidden layer NN with ReLU and 30 neurons)
        self.fc1 = nn.Linear(in_features=input_size, out_features=30)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features=30, out_features=1)  # Assuming binary classification

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return torch.sigmoid(x)

class Model4(nn.Module):
    def __init__(self):
        super(Model4, self).__init__()
        # Define your model architecture (Two-hidden layer NN with ReLU)
        self.fc1 = nn.Linear(in_features=input_size, out_features=10)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features=10, out_features=10)
        self.fc3 = nn.Linear(in_features=10, out_features=1)  # Assuming binary classification

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return torch.sigmoid(x)



# Function to train models using different optimization algorithms
def train_model(model, optimizer, criterion, X_train_tensor, y_train_tensor, num_epochs=100, batch_size=10):
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()  # Zero the gradients
    
        outputs = model(X_train_tensor)  # Forward pass
        loss = criterion(outputs, y_train_tensor)  # Calculate the loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights

# Function to evaluate models
def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        predicted = (outputs.squeeze() > 0.5).float()
        accuracy = (predicted == y_test.float()).float().mean().item()
        error_rate = 1 - accuracy
        return error_rate


#logistic regression
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
    max_iter = 1000
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
        
    return errors #end



filename = "spambase.data"
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


# Define parameters
train_percentages = [10, 20, 30]
num_splits = 100
batch_size = 10
num_epochs = 1
input_size = X.shape[1]  # Assuming input data shape

# Initialize lists to store results
mean_errors_sgd = {i: [] for i in range(1, 5)}  # Dictionary to store mean errors for each model
std_errors_sgd = {i: [] for i in range(1, 5)}   # Dictionary to store standard deviations for each model
mean_errors_adam = {i: [] for i in range(1, 5)}  # Dictionary to store mean errors for each model
std_errors_adam = {i: [] for i in range(1, 5)}   # Dictionary to store standard deviations for each model

# Iterate over models
for model_idx in range(1, 9):
    #start logistic regression
    for train_percent in train_percentages: #loops through train sizes
        error_per_split_adam = []
        error_per_split_sgd = []
        for split in range(num_splits):
            
            # Split data into train and test sets
            indices = np.random.permutation(len(X))
            train_size = int(0.8 * len(X)) #80% train
            actual_train_size = math.floor(train_percent/100 * len(X)) #use input to get the necessary percentage of training data used
            train_indices, test_indices = indices[:actual_train_size], indices[train_size:]
    
            X_train, X_test = torch.tensor(X[train_indices], dtype=torch.float), torch.tensor(X[test_indices], dtype=torch.float)
            y_train, y_test = torch.tensor(y[train_indices], dtype=torch.float), torch.tensor(y[test_indices], dtype=torch.float)
    
            if model_idx==1:
                model = Model1()
            elif model_idx==2:
                model = Model1()
            elif model_idx==3:
                model = Model2()
            elif model_idx==4:
                model = Model2()  
            elif model_idx==5:
                model = Model3()
            elif model_idx==6:
                model = Model3()
            elif model_idx==7:
                model = Model4()
            elif model_idx==8:
                model = Model4()
    
            criterion = nn.BCELoss()
            optimizer_sgd = optim.SGD(model.parameters(), lr=0.01)
            optimizer_adam = optim.Adam(model.parameters(), lr=0.01)
    
            # Train models
            train_model(model, optimizer_sgd if model_idx%2 == 1 else optimizer_adam, criterion, X_train, y_train, num_epochs, batch_size)
    
            # Evaluate models
            error_sgd = evaluate_model(model, X_test, y_test)
            if model_idx%2 == 1:
                error_per_split_sgd.append(error_sgd)
            else:
                error_per_split_adam.append(error_sgd)
    
        mean_errors_sgd[(model_idx+1)//2].append(np.mean(error_per_split_sgd))  # Mean test error for SGD
        mean_errors_adam[(model_idx+1)//2].append(np.mean(error_per_split_adam))  # Mean test error for Adam
    
        std_errors_sgd[(model_idx+1)//2].append(np.std(error_per_split_sgd))  # Standard deviation for SGD
        std_errors_adam[(model_idx+1)//2].append(np.std(error_per_split_adam))  # Standard deviation for Adam

#clean output
cleaned_mean_errors_sgd = {}
    # loop through each key in the original dict
for key_name in mean_errors_sgd:
    # for each item in the values for this key, if it is not nan then add to new list
    items_cleaned = [item for item in mean_errors_sgd[key_name] if not np.isnan(item)]
    # add this list of cleaned items to the cleaned dictionary
    cleaned_mean_errors_sgd[key_name] = items_cleaned
cleaned_std_errors_sgd = {}
    # loop through each key in the original dict
for key_name in std_errors_sgd:
    # for each item in the values for this key, if it is not nan then add to new list
    items_cleaned_2 = [item for item in std_errors_sgd[key_name] if not np.isnan(item)]
    # add this list of cleaned items to the cleaned dictionary
    cleaned_std_errors_sgd[key_name] = items_cleaned_2
cleaned_mean_errors_adam = {}
    # loop through each key in the original dict
for key_name in mean_errors_adam:
    # for each item in the values for this key, if it is not nan then add to new list
    items_cleaned3 = [item for item in mean_errors_adam[key_name] if not np.isnan(item)]
    # add this list of cleaned items to the cleaned dictionary
    cleaned_mean_errors_adam[key_name] = items_cleaned3
cleaned_std_errors_adam = {}
    # loop through each key in the original dict
for key_name in std_errors_adam:
    # for each item in the values for this key, if it is not nan then add to new list
    items_cleaned_4 = [item for item in std_errors_adam[key_name] if not np.isnan(item)]
    # add this list of cleaned items to the cleaned dictionary
    cleaned_std_errors_adam[key_name] = items_cleaned_4

# Plotting
models = ['Model 1', 'Model 2', 'Model 3', 'Model 4', 'Model 5']
lrErrors = logisticRegression(filename, 100, [10, 20, 30])

#plot Training Percent vs. Mean Prediction Accuracy
means = [i[0] for i in lrErrors]
stdevs = [i[1] for i in lrErrors]

plt.style.use('ggplot')
plt.rc('axes', labelsize = 20)
plt.rc('xtick', labelsize = 20)
plt.rc('ytick', labelsize = 20)
plt.rc('legend', fontsize = 18)
plt.rc('lines', linewidth = 4)

plt.errorbar(x = train_percentages, y = means, yerr = stdevs)
        
plt.legend()
plt.ylim(0.37,0.43)
plt.xlabel("Training Percent", fontsize=14)
plt.ylabel("Mean Prediction Accuracy", fontsize=14)
plt.title("Training Percent vs. Mean Prediction Accuracy - Logistic Regression", fontsize=24)
plt.show()

for model_idx in range(1,5):
    plt.errorbar(train_percentages, cleaned_mean_errors_sgd[model_idx], yerr=cleaned_std_errors_sgd[model_idx])
    plt.xlabel('Percentage of Training Set')
    plt.ylabel('Mean Test Error Rate')
    plt.title('SGD Comparison of Accuracy for Model {}'.format(model_idx))
    plt.legend()
    plt.show()
for model_idx in range(1,5):
    plt.errorbar(train_percentages, cleaned_mean_errors_adam[model_idx], yerr=cleaned_mean_errors_adam[model_idx])
    plt.xlabel('Percentage of Training Set')
    plt.ylabel('Mean Test Error Rate')
    plt.title('Adam Comparison of Accuracy for Model {}'.format(model_idx))
    plt.legend()
    plt.show()
