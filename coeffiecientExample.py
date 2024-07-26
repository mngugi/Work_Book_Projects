import numpy as np
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# Data
input_1 = np.array([130, 250, 190, 300, 210, 220, 170]).reshape(-1, 1)
input_2 = np.array([1900, 2600, 2200, 2900, 2400, 2300, 2100]).reshape(-1, 1)
output = np.array([16.3, 10.2, 11.1, 7.1, 12.1, 13.2, 14.2])

# Combine inputs into a single matrix
X = np.hstack((input_1, input_2))

# Create and fit the model
model = LinearRegression()
model.fit(X, output)

# Get the coefficients and bias
coefficients = model.coef_
bias = model.intercept_

coefficients, bias
