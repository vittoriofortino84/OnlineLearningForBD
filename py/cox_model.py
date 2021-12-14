from abc import abstractmethod, ABC

import matplotlib
from lifelines import CoxPHFitter
# from sksurv.linear_model import CoxPHSurvivalAnalysis
from matplotlib import pyplot as plt

from data_utils import merge_x_y
from py.cox_plots import show_cumulative_hazard_functions


class CoxPredictor(ABC):

    @abstractmethod
    def score(self, x_test, y_test) -> float:
        raise NotImplementedError()

    @abstractmethod
    def p_vals(self):
        raise NotImplementedError()

    @abstractmethod
    def params(self):
        raise NotImplementedError()


class CoxModel(ABC):

    @abstractmethod
    def fit_estimator(self, x_train, y_train, alpha: float = 0) -> CoxPredictor:
        raise NotImplementedError()


# class SKSurvCoxPredictor(CoxPredictor):
#
#     __estimator: CoxPHSurvivalAnalysis
#
#     def __init__(self, estimator: CoxPHSurvivalAnalysis):
#         self.__estimator = estimator
#
#     def score(self, x_test, y_test) -> float:
#         return self.__estimator.score(x_test, y_test)
#
#     def p_vals(self):
#         raise NotImplementedError()
#
#     def params(self):
#         raise NotImplementedError()
#
#
# class SKSurvCoxModel(CoxModel):
#
#     def fit_estimator(self, x_train, y_train, alpha: float = 0) -> CoxPredictor:
#         estimator = CoxPHSurvivalAnalysis(alpha=alpha).fit(x_train, y_train)
#         return SKSurvCoxPredictor(estimator)


class LifelinesCoxPredictor(CoxPredictor):
    __estimator: CoxPHFitter

    def __init__(self, estimator: CoxPHFitter):
        self.__estimator = estimator

    @staticmethod
    def merge_x_y(x, y):
        return merge_x_y(x=x, y=y)

    def score(self, x_test, y_test) -> float:
        df = self.merge_x_y(x=x_test, y=y_test)
        return self.__estimator.score(df, scoring_method="concordance_index")

    def p_vals(self):
        summary = self.__estimator.summary
        return summary['p']

    def params(self):
        return self.__estimator.params_


class LifelinesCoxModel(CoxModel):

    def fit_estimator(self, x_train, y_train, alpha: float = 0) -> CoxPredictor:
        df = LifelinesCoxPredictor.merge_x_y(x=x_train, y=y_train)
        if df.isnull().values.any():
            print("Nulls detected in the dataframe")
            print(df.isnull())
        estimator = CoxPHFitter(penalizer=alpha, l1_ratio=0).fit(df=df, duration_col='time', event_col='event')
        return LifelinesCoxPredictor(estimator)




