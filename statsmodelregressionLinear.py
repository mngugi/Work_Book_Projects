import statsmodels.api as sm    

import numpy as np

duncan_prestige = sm.datasets.get_rdataset("Duncan", "carData")
Y = duncan_prestige.data['income']
X = duncan_prestige.data['education']
X = sm.add_constant(X)
model = sm.OLS(Y,X)
results = model.fit()
print(results.params)
print('\n--------------------------\n')

results_ = results.tvalues
print(results_)

print('\n--------------------------\n')
print(results.t_test([1, 0]))

print('\n--------------------------\n')
print(results.f_test(np.identity(2)))



