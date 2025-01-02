from statsmodels.datasets.longley import load_pandas

y = load_pandas().endog
X = load_pandas().exog
X = sm.add_constant(X)

ols_model = sm.OLS(y, X)
ols_results = ols_model.fit()
print(ols_results.summary())
