import statistics

from cox_model import CoxModel, LifelinesCoxModel
from univariate_analysis import univariate_analysis


def risk_score(coefs: dict, sample: dict) -> float:
    res = 0.0
    for key in coefs:
        if key in sample:
            res += coefs[key]*sample[key]  # TODO Can benefit from stable sum
    return res


def risk_scores(coefs: dict, x) -> [float]:
    x_dict = x.to_dict(orient='records')
    res = [risk_score(coefs=coefs, sample=i) for i in x_dict]
    return res


def prognostic_coefficients(x, y, model: CoxModel = LifelinesCoxModel(), alpha=0.0, p_val=0.05):
    uni_res = univariate_analysis(x=x, y=y, model=model, alpha=alpha)
    uni_res_list = [(f, s, p) for f, s, p in zip(uni_res['feature'], uni_res['score'], uni_res['p_val'])]
    res = {}
    for r in uni_res_list:
        if r[2] < p_val:
            res[r[0]] = r[1]
    return res


class RSPrognosticClassifier:
    __coeffs: dict
    __cutoff: float

    def __init__(self, coeffs: dict, cutoff: float):
        self.__coeffs = coeffs
        self.__cutoff = cutoff

    def predict(self, x) -> [int]:
        scores = risk_scores(self.__coeffs, x=x)
        res = [s <= self.__cutoff for s in scores]
        return res


class RSPrognosticFitter:

    def fit(self, x_train, y_train, alpha=0.0, p_val=0.05) -> RSPrognosticClassifier:
        coeffs = prognostic_coefficients(x=x_train, y=y_train, alpha=alpha, p_val=p_val)
        scores = risk_scores(coeffs, x_train)
        cutoff = statistics.median(scores)
        return RSPrognosticClassifier(coeffs=coeffs, cutoff=cutoff)
