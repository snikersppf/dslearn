import pandas as pd
import numpy as np

np.random.seed(67)

exp = np.array([0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10])

salary = 30 + 5.5 * exp + np.random.normal(0, 2, size=len(exp))

exp_mean, exp_std = exp.mean(), exp.std()
exp_norm = (exp - exp_mean) / exp_std

w = 0.0
b = 0.0
alpha = 0.01
n_epochs = 1000
n = len(exp_norm)

losses = []

for epoch in range(n_epochs):
    salary_pred = w * exp_norm + b
    
    error = salary - salary_pred
    mse = (error ** 2).mean()
    losses.append(mse)
    
    dw = (-2/n) * (exp_norm * error).sum()
    db = (-2/n) * error.sum()
    
    w -= alpha * dw
    b -= alpha * db

def predict(exp_new):
    exp_norm = (exp_new - exp_mean) / exp_std
    return w * exp_norm + b

inp_sal = int(input())
print(f"expected salary: {predict(inp_sal)}")

    