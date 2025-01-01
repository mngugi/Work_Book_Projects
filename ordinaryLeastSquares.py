a = '''
Ordinary Least Squares (OLS) in Linear Regression

Ordinary Least Squares (OLS) is a statistical method used to estimate the parameters
(or coefficients) of a linear regression model. In simple terms, OLS helps us find the
"best-fitting" line that describes the relationship between the independent variable(s)
and the dependent variable by minimizing the sum of squared differences between the observed
data and the predicted values.

Example of OLS in Action

Let's consider a simple example where we want to predict a person's weight YY based on their
height XX. We collect data and fit an OLS model. The goal is to find the best line (in the 
form Y=β0+β1XY=β0​+β1​X) that minimizes the sum of squared errors between the observed weights 
and the weights predicted by the model.

After fitting the model, the OLS estimates for β0β0​ and β1β1​ will give us the equation of 
the best-fitting line, which we can use to predict the weight for any given height.

'''
print(a)
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
# model intercept 
X = sm.add_constant(X)
y = np.dot(X, beta) + e

# fit summary
model = sm.OLS(y, X)
results = model.fit()
print(results.summary())

print("Parameters: ", results.params)
print("R2: ", results.rsquared)

nsample = 50
sig = 0.5
x = np.linspace(0, 20, nsample)
X = np.column_stack((x, np.sin(x), (x - 5) ** 2, np.ones(nsample)))
beta = [0.5, 0.5, -0.02, 5.0]

y_true = np.dot(X, beta)
y = y_true + sig * np.random.normal(size=nsample)

res = sm.OLS(y, X).fit()
print(res.summary())

print("Parameters: ", res.params)
print("Standard errors: ", res.bse)
print("Predicted values: ", res.predict())

pred_ols = res.get_prediction()
iv_l = pred_ols.summary_frame()["obs_ci_lower"]
iv_u = pred_ols.summary_frame()["obs_ci_upper"]

fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(x, y, "o", label="data")
ax.plot(x, y_true, "b-", label="True")
ax.plot(x, res.fittedvalues, "r--.", label="OLS")
ax.plot(x, iv_u, "r--")
ax.plot(x, iv_l, "r--")
ax.legend(loc="best")

fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(x, y, "o", label="data")
ax.plot(x, y_true, "b-", label="True")
ax.plot(x, res.fittedvalues, "r--.", label="OLS")
ax.plot(x, iv_u, "r--")
ax.plot(x, iv_l, "r--")
ax.legend(loc="best")

# Save the plot as an image file (e.g., PNG)
plt.savefig("ols_graph.png", dpi=300, bbox_inches='tight')  # Save with high resolution

# Show the plot
plt.show()

