import pandas as pd
import numpy as np
import tqdm
import math
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import BaggingRegressor


def impute_missing_values(df_, var_deviation_tolerance=0.97, actual_or_gaussian_residuals='actual',
                          col_floor_ceiling_dict=None, scores=False):
    """Impute missing values while minimizing distortion of variable distribution
    by creating a bagged model using other variables and adding residuals to output values

    Parameters:
    df: dataframe with missing values
    var_deviation_tolerance: target percent deviation from original variable distributions
    actual_or_guassian_residuals: apply residuals to model outputs from actual distribution or from
        a gaussian distribution based on residuals' means and variances
    col_floor_ceiling_dict: a dictionary with the variable name and a tuple of the min and max for variables
        with a finite range. Use float(inf) or float(-inf) for variables that are limited in only one direction
    scores: return accuracy score of models per variable

    Returns:
    df: df with imputed values
    problems: columns that failed to impute
    column_scores: accuracy scores of imputation model on non-missing values
    regressors: trained regressors that can be used for imputing missing values in future production data
    """
    regressors = []  # List that will contain the regressor trained for predicting values for each variable

    df_ = df_.copy()
    columns = df_.columns
    type_dict = df_.dtypes.to_dict()
    missing_columns = list(df_.isna().sum()[df_.isna().sum() > 0].sort_values().index)
    have_columns = [i for i in columns if i not in missing_columns]
    column_scores = {}
    problems = []
    for col in tqdm.tqdm(missing_columns):
        # noinspection PyBroadException
        try:
            percent_missing = df_[col].isna().sum() / df_.shape[0]
            m = math.ceil(percent_missing / ((1 / var_deviation_tolerance) - 1))
            other_columns = [i for i in columns if i != col]
            na_index = df_[df_[col].isna() == 1].index
            have_index = [i for i in df_.index if i not in na_index]
            na_have_cols = set(df_.loc[na_index, other_columns].dropna(axis=1).columns)
            have_have_cols = set(df_.loc[have_index, other_columns].dropna(axis=1).columns)
            both_cols = na_have_cols.intersection(have_have_cols)
            int_df = pd.get_dummies(df_.loc[:, both_cols], drop_first=False)
            X_have = int_df.loc[have_index, :]
            y_have = df_[col][have_index]
            X_na = int_df.loc[na_index, :]

            if type_dict[col] == 'object':
                le = LabelEncoder()
                y_have = le.fit_transform(y_have)
                df_[col][have_index] = y_have
                rf = RandomForestClassifier()
                bag_classifier = BaggingClassifier(base_estimator=rf, n_estimators=m)
                bag_classifier.fit(X_have, y_have)
                column_scores[col] = bag_classifier.score(X_have, y_have)
                residual_preds = bag_classifier.predict(X_have)
                residuals = y_have - residual_preds
                preds = bag_classifier.predict(X_na)
                regressors.append(bag_classifier)
            else:
                bag_regressor = BaggingRegressor(n_estimators=m)
                bag_regressor.fit(X_have, y_have)
                column_scores[col] = bag_regressor.score(X_have, y_have)
                residual_preds = bag_regressor.predict(X_have)
                residuals = y_have - residual_preds
                preds = bag_regressor.predict(X_na)
                regressors.append(bag_regressor)
            if actual_or_gaussian_residuals == 'actual':
                rand_residuals = np.random.choice(residuals, len(X_na), replace=True)
            else:
                rand_residuals = np.random.normal(residuals.mean(), residuals.std(), len(X_na))
            preds = preds + rand_residuals
            if type_dict[col] == 'object':
                preds = preds.round()
            if not col_floor_ceiling_dict is None:
                if col in col_floor_ceiling_dict.keys():
                    preds = np.clip(preds, col_floor_ceiling_dict[col][0], col_floor_ceiling_dict[col][1])
            df_[col][na_index] = preds
            have_columns.append(col)
        except:
            problems.append(col)
    if not scores:
        return df_, problems, regressors
    else:
        return df_, problems, column_scores, regressors


def change_type(df_: pd.DataFrame, colname: str, astype: type) -> pd.DataFrame:
    df_[colname] = df_[colname].astype(astype)
    if df_[colname].max() == np.inf:
        print(f'{colname}: Value out of range for type {astype}')
    else:
        return df_
