import statistics

import pandas as pd

from cox_model import LifelinesCoxModel, CoxPredictor, CoxModel, LifelinesCoxPredictor
from risk_score import RSPrognosticClassifier, risk_scores, prognostic_coefficients


class RiskScoreCoxPredictor(CoxPredictor):

    __rs_prognostic_classifier: RSPrognosticClassifier
    __lifelines_predictor: LifelinesCoxPredictor

    def __init__(self, rs_prognostic_classifier: RSPrognosticClassifier, lifelines_predictor: LifelinesCoxPredictor):
        self.__rs_prognostic_classifier = rs_prognostic_classifier
        self.__lifelines_predictor = lifelines_predictor

    def score(self, x_test, y_test) -> float:
        rs_classes = self.__rs_prognostic_classifier.predict(x=x_test)
        df_classes = pd.DataFrame()
        df_classes["risk_group"] = rs_classes
        return self.__lifelines_predictor.score(x_test=df_classes, y_test=y_test)

    def p_vals(self):
        return self.__lifelines_predictor.p_vals()

    def params(self):
        return self.__lifelines_predictor.params()


class RiskScoreCoxModel(CoxModel):
    __p_val: float

    def __init__(self, p_val=0.05):
        self.__p_val = p_val

    def fit_estimator(self, x_train, y_train, alpha: float = 0) -> CoxPredictor:
        coeffs = prognostic_coefficients(x=x_train, y=y_train, alpha=alpha, p_val=self.__p_val)
        scores = risk_scores(coeffs, x_train)
        cutoff = statistics.median(scores)
        rs_prognostic_classifier = RSPrognosticClassifier(coeffs=coeffs, cutoff=cutoff)
        rs_classes = rs_prognostic_classifier.predict(x=x_train)
        df_classes = pd.DataFrame()
        df_classes["risk_group"] = rs_classes
        lifelines_predictor = LifelinesCoxModel().fit_estimator(x_train=df_classes, y_train=y_train, alpha=alpha)
        return RiskScoreCoxPredictor(
            rs_prognostic_classifier=rs_prognostic_classifier, lifelines_predictor=lifelines_predictor)
