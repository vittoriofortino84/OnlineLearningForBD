import pandas as pd

from cox_model import CoxModel, LifelinesCoxModel


def univariate_analysis(x, y, model: CoxModel = LifelinesCoxModel(), alpha=0.0):
    res = pd.DataFrame(columns=['feature', 'score', 'p_val', 'coefficient'])
    pos = 0
    for feat_name in x:
        feat_df = x[[feat_name]]
        feat_predictor = model.fit_estimator(x_train=feat_df, y_train=y, alpha=alpha)
        score = feat_predictor.score(x_test=feat_df, y_test=y)
        p_val = feat_predictor.p_vals()[0]
        coefficient = feat_predictor.params()[feat_name]
        res.loc[pos] = [feat_name, score, p_val, coefficient]
        pos += 1
    res.sort_values(by=['p_val'], inplace=True, ignore_index=True)
    return res


def univariate_analysis_with_covariates(x, y, cov, model: CoxModel = LifelinesCoxModel(), alpha=0.0):
    res = pd.DataFrame(columns=['feature', 'score', 'p_val', 'coefficient'])
    pos = 0
    for feat_name in x:
        feat_df = pd.concat(objs=[cov, x[[feat_name]]], axis=1)
        feat_predictor = model.fit_estimator(x_train=feat_df, y_train=y, alpha=alpha)
        score = feat_predictor.score(x_test=feat_df, y_test=y)
        p_val = feat_predictor.p_vals()[feat_name]
        coefficient = feat_predictor.params()[feat_name]
        res.loc[pos] = [feat_name, score, p_val, coefficient]
        pos += 1
    res.sort_values(by=['p_val'], inplace=True, ignore_index=True)
    return res


