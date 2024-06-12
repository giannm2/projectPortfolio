# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 23:10:38 2023

@author: 2mgia
"""

import numpy as np
# import matplotlib.pyplot as plt
import time


def pegasos_svm(k, num_runs):
    # Pegasos SVM implementation
    # k: Mini-batch size
    # num_runs: Number of runs for averaging
    
    # Load your dataset here 
    X = np.array([[1, 1], [2, 1], [2, 2], [4, 5], [6, 5], [5, 7], [5, 1], [7, 7]])
    y = np.array([1, 1, 1, -1, -1, -1, -1, -1])
    
    n, d = X.shape
    w = np.zeros(d)  # Initialize weight vector
    lambda_reg = 1e-6  # Regularization parameter
    num_iterations = 750  # Adjust as needed
    runtimes = []
    objective_values = np.zeros((num_runs, num_iterations))

    for run in range(num_runs):
        start_time = time.time()
        for t in range(1, num_iterations + 1):
            indices = np.random.choice(n, k, replace=False)
            X_t = X[indices, :]
            y_t = y[indices]
            
            eta = 1 / (lambda_reg * t)
            
            # Compute sub-gradient
            subgrad = np.zeros(d)
            for i in range(k):
                if y_t[i] * np.dot(w, X_t[i, :]) < 1:
                    subgrad += -y_t[i] * X_t[i, :]
            
            # Update weight vector
            w = (1 - eta * lambda_reg) * w + (eta / k) * subgrad
            
            # Compute and store objective value
            objective_values[run, t - 1] = 0.5 * np.dot(w, w) + \
                                           lambda_reg * np.mean(np.maximum(1 - y * np.dot(X, w), 0))

        end_time = time.time()
        runtime = end_time - start_time
        runtimes.append(runtime)

    avg_runtime = np.mean(runtimes)
    std_dev_runtime = np.std(runtimes, ddof=1)

    return avg_runtime, std_dev_runtime, objective_values

def mysgdsvm(filename, k, num_runs):
    # Main function to load data, run Pegasos SVM, and display results
    avg_runtime, std_dev_runtime, objective_values = pegasos_svm(filename, k, num_runs)
    
    print(f"Average Runtime: {avg_runtime:.4f} seconds")
    print(f"Standard Deviation of Runtime: {std_dev_runtime:.4f} seconds")

    # # Plotting code for objective values
    # for run in range(num_runs):
    #     plt.plot(objective_values[run, :], label=f"Run {run + 1}")

    # plt.title(f"Mini-batch Size: {k}")
    # plt.xlabel("Iterations")
    # plt.ylabel("Primal Objective Function Value")
    # plt.legend()
    # plt.show()
    # return objective_values

# Example usage
# filename = 'C:/Users/2mgia/Dropbox/MNIST-13.csv'
# k_values = [1, 20, 100, 200, 2000]
# num_runs = 5

# for k in k_values:
#     print(f"\nRunning Pegasos for k = {k}")
#     mysgdsvm(filename, k, num_runs)
