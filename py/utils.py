from statistics import mean

import sklearn
from sksurv.linear_model import CoxPHSurvivalAnalysis


def create_folds(x, y, n_folds: int = 10, seed=4985):
    skf = sklearn.model_selection.StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    res = []
    y_event = []
    for i in y:
        y_event.append(i[0])
    for train_index, test_index in skf.split(X=x, y=y_event):
        res.append([train_index, test_index])
    return res


def train_test_one_fold(x_train, y_train, x_test, y_test, alpha=0):
    estimator = CoxPHSurvivalAnalysis(alpha=alpha).fit(x_train, y_train)
    score = estimator.score(x_test, y_test)
    return score


def cross_validate(x, y, n_folds: int = 10, alpha=0, seed=78245):
    folds = create_folds(x, y, n_folds=n_folds, seed=seed)
    scores = []
    for train_index, test_index in folds:
        x_train = x.iloc[train_index]
        x_test = x.iloc[test_index]
        y_train = y[train_index]
        y_test = y[test_index]
        fold_score = train_test_one_fold(x_train, y_train, x_test, y_test, alpha)
        scores.append(fold_score)
    return mean(scores)
