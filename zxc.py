import numpy as np

degrees = np.random.randint(-10, 30, 150)
time = np.random.randint(0, 12, 150)

y = 5 + 0.5 * degrees + 2.0 * time + np.random.normal(0,2, size = len(degrees))

degrees_mean, degrees_std = degrees.mean(), degrees.std()
degrees_norm = (degrees - degrees_mean) / degrees_std

time_mean, time_std = time.mean(), time.std()
time_norm = (time - time_mean) / time_std

X = np.column_stack([degrees_norm, time_norm])

w = np.zeros(2)
b = 0.0
alpha = 0.01
n_epochs = 1000
n = len(degrees_norm)

losses = []

for epoch in range(n_epochs):
    y_pred = X @ w + b
    
    error = y - y_pred
    mse = (error ** 2).mean()
    
    losses.append(mse)
    
    dw = (-2 / n) * (X.T @ error)
    db = (-2 / n) * error.sum()
    
    w -= alpha * dw
    b -= alpha * db
    
def predict(degreees_new, time_new):
    degrees_new_norm  = (degreees_new - degrees_mean) / degrees_std
    time_new_norm = (time_new - time_mean) / time_std
    
    return w @ [degrees_new_norm, time_new_norm] + b

print("input degrees:")
inp_degrees = int(input())
print("input time: ")
inp_time = int(input())


print(f"expected y: {predict(inp_degrees, inp_time)} ")
     