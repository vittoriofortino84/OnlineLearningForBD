from matplotlib import pyplot as plt


def show_cumulative_hazard_functions(estimator, X, num_rows=10):
    chf_funcs = estimator.predict_cumulative_hazard_function(X.iloc[:num_rows])
    for fn in chf_funcs:
        plt.step(fn.x, fn(fn.x), where="post")
    plt.ylim(0, 2)
    plt.show()


def show_survival_functions(estimator, X, num_rows=10):
    chf_funcs = estimator.predict_survival_function(X.iloc[:num_rows])
    for fn in chf_funcs:
        plt.step(fn.x, fn(fn.x), where="post")
    plt.ylim(0, 1)
    plt.show()
