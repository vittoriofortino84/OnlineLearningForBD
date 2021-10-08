from cox_model import CoxModel, LifelinesCoxModel


def univariate_analysis(x, y, model: CoxModel = LifelinesCoxModel(), n_folds=10, alpha=0.0):

    for feat_name in x:
        feat_df = x[[feat_name]]
        feat_predictor = model.fit_estimator(x_train=feat_df, y_train=y, alpha=alpha)
