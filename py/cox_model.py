from abc import abstractmethod, ABC
from lifelines import CoxPHFitter
from sksurv.linear_model import CoxPHSurvivalAnalysis

from data_utils import merge_x_y


class CoxPredictor(ABC):

    @abstractmethod
    def score(self, x_test, y_test) -> float:
        raise NotImplementedError()

    @abstractmethod
    def p_vals(self):
        raise NotImplementedError()


class CoxModel(ABC):

    @abstractmethod
    def fit_estimator(self, x_train, y_train, alpha: float = 0) -> CoxPredictor:
        raise NotImplementedError()


class SKSurvCoxPredictor(CoxPredictor):

    __estimator: CoxPHSurvivalAnalysis

    def __init__(self, estimator: CoxPHSurvivalAnalysis):
        self.__estimator = estimator

    def score(self, x_test, y_test) -> float:
        return self.__estimator.score(x_test, y_test)

    def p_vals(self):
        raise NotImplementedError()


class SKSurvCoxModel(CoxModel):

    def fit_estimator(self, x_train, y_train, alpha: float = 0) -> CoxPredictor:
        estimator = CoxPHSurvivalAnalysis(alpha=alpha).fit(x_train, y_train)
        return SKSurvCoxPredictor(estimator)


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


class LifelinesCoxModel(CoxModel):

    def fit_estimator(self, x_train, y_train, alpha: float = 0) -> CoxPredictor:
        df = LifelinesCoxPredictor.merge_x_y(x=x_train, y=y_train)
        if df.isnull().values.any():
            print(df.isnull())
        estimator = CoxPHFitter(penalizer=alpha, l1_ratio=0).fit(df=df, duration_col='time', event_col='event')
        return LifelinesCoxPredictor(estimator)




