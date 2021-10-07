import matplotlib.pyplot as plt
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.datasets import load_whas500

X, y = load_whas500()

X = X.astype(float)

print(X.head())
print(y[1:20])

estimator = CoxPHSurvivalAnalysis().fit(X, y)

chf_funcs = estimator.predict_cumulative_hazard_function(X.iloc[:10])

for fn in chf_funcs:
    plt.step(fn.x, fn(fn.x), where="post")
plt.ylim(0, 2)
plt.show()

surv_funcs = estimator.predict_survival_function(X.iloc[:10])

for fn in surv_funcs:
    plt.step(fn.x, fn(fn.x), where="post")

plt.ylim(0, 1)
plt.show()
