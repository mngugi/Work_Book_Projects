import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

np.random.seed(9876789)

# OLS estimation

nsample = 100
x = np.linspace(0, 10, 100)
X = np.column_stack((x, x ** 2))
beta = np.array([1, 0.1, 10])
e = np.random.normal(size=nsample)
# odel intercept 
X = sm.add_constant(X)
y = np.dot(X, beta) + e

# fit summary
model = sm.OLS(y, X)
results = model.fit()
print(results.summary())
