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

# fit summary 
res2 = sm.OLS(y, X).fit()
print(res2.summary())


# f- test 

R = [[0, 1, 0, 0], [0, 0, 1, 0]]
print(np.array(R))
print(res2.f_test(R))

b= '''
2. The Array: [[0 1 0 0], [0 0 1 0]]

This is likely a contrast matrix or design matrix used in the model to define certain relationships
or hypotheses for testing (e.g., in the F-test). Here's what it likely represents:

    Each row corresponds to a linear combination of coefficients being tested.
    For example:
        Row 1: [0 1 0 0] means the second coefficient is being tested (e.g., testing if β2=0β2​=0).
        Row 2: [0 0 1 0] means the third coefficient is being tested (e.g., testing if β3=0β3​=0).

In this case, the F-test will assess whether the specified coefficients (e.g., β2β2​ and β3β3​) are jointly significant.
'''
print(b) 
c= '''
3. F-Test Information

    <F test: F=34.455088527508664, p=7.164481588009625e-10, df_denom=46, df_num=2>

This describes the results of an F-test, which is used to evaluate whether a group of coefficients
in the regression model is jointly significant. Here's what the components mean:

    F-statistic (F): 34.455
        This value measures the ratio of the explained variance to the unexplained variance for the
        group of coefficients being tested. A higher F-statistic suggests stronger evidence against the null hypothesis.
    p-value (p): 7.164e-10
        The p-value indicates the probability of observing the F-statistic (or something more extreme) under the null 
        hypothesis. A very small p-value (like 7.16×10−107.16×10−10) suggests that the null hypothesis is highly unlikely.
        In this case, the p-value is much smaller than common significance levels like α=0.05α=0.05, so we reject the null
        hypothesis.
    Degrees of freedom (df_num, df_denom):
        df_num (2): The number of parameters being tested (in this case, 2 coefficients).
        df_denom (46): The degrees of freedom in the residuals, which is the number of observations minus the number 
        of estimated parameters.
        
'''

d = '''
What Does This Mean?

    The F-Test: The test is evaluating whether the two coefficients corresponding to [0 1 0 0] and [0 0 1 0] are jointly equal
    to 0 (i.e., testing their significance in explaining the dependent variable).
        Null Hypothesis: β2=β3=0β2​=β3​=0 (the coefficients being tested have no impact on the dependent variable).
        Alternative Hypothesis: At least one of the coefficients is non-zero (i.e., at least one contributes significantly 
        to explaining the dependent variable).

    Interpretation:
        The very large F-statistic (34.455) and extremely small p-value (7.164e-10) suggest strong evidence against the null
        hypothesis. This means that at least one of the coefficients (β2β2​ or β3β3​) is statistically significant in explaining
        the dependent variable.

    Practical Implication:
        If these coefficients represent predictors in a regression model, they are important for understanding the dependent
        variable and should be included in the model.
'''
print(d) 

print(res2.f_test("x2 = x3 = 0"))

beta = [1.0, 0.3, -0.0, 10]
y_true = np.dot(X, beta)
y = y_true + np.random.normal(size=nsample)

res3 = sm.OLS(y, X).fit()
