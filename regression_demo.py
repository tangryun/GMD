#! /bin/env python
#-*- coding:utf-8 -*-

__author__='Rongyun Tang'
import time
start = time.time()

import os, glob, pdb
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
# from sklearn.feature_selection import RFE
from sklearn.metrics import mean_squared_error, mean_absolute_error,\
    explained_variance_score, r2_score, mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression, SGDRegressor, BayesianRidge, ElasticNet, LogisticRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, \
    GradientBoostingRegressor, StackingRegressor, BaggingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost.sklearn import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
import openpyxl
import shutil

end3 = time.time()
print('block 3, using time: ', end3-start)

from openpyxl import load_workbook
import warnings
warnings.filterwarnings('ignore')
random_seed = 1
print('librarys are loaded!')

# Subset regression data according to best-performed classification model (RF)
def read_classification_error(path_error):
    file_err = path_error + 'classification_accuracy_summary.xlsx'
    err_data = pd.read_excel(file_err)
    PPV = err_data[(err_data.Model=='RF') & (err_data.Type == 'testing')].PPV.values[0]
    FDR = err_data[(err_data.Model=='RF') & (err_data.Type == 'testing')].FDR.values[0]
    FOR = err_data[(err_data.Model=='RF') & (err_data.Type == 'testing')].FOR.values[0]
    NPV = err_data[(err_data.Model=='RF') & (err_data.Type == 'testing')].NPV.values[0]
    return FDR, FOR, NPV, PPV


def create_directories(path_file):
    with open(path_file, "r") as file:
        lines = file.readlines()
        path_input = lines[0].strip().split(": ")[1]
        input_location = lines[1].strip().split(": ")[1]
        path_error = lines[2].strip().split(": ")[1]
    path_output_train = path_input + '01_regression_train/'
    path_output_test = path_input + '01_regression_test/'
    directories = [path_output_train, path_output_test]

    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print('create directory: ', directory)

    return path_input, input_location, path_error, path_output_train, path_output_test


def metrics(true, preds):
    """
    Calculates evaluation metrics.

    Parameters:
    true (array-like): The true values.
    preds (array-like): The predicted values.

    Returns:
    A list of evaluation metrics: [MSE, MAE, VAR, R2, MAPE].
    """
    # regression metrics
    MSE = mean_squared_error(true, preds)
    MAE = mean_absolute_error(true, preds)
    VAR = explained_variance_score(true, preds)
    T2 = pd.DataFrame()
    T2['True1'] = true
    T2['Pred1'] = preds
    T2 = T2.loc[(T2 != 0).any(axis=1)]
    true1 = T2.True1.tolist()
    preds1 = T2.Pred1.tolist()
    R2 = r2_score(true1, preds1)
    MAPE = mean_absolute_percentage_error(true, preds)
    score_df = [MSE, MAE, VAR, R2, MAPE]
    return score_df


def run_exps_seasons(month: str,
                     path_error,
                     writer, writer_all,
                     X_train_r: pd.DataFrame, y_train_r: pd.DataFrame,
                     X_test_r: pd.DataFrame, y_test_r: pd.DataFrame,
                     train_ind_r: list, test_ind_r: list,
                     X_test_0_r: pd.DataFrame, y_test_0_r: pd.DataFrame,
                     test_ind_0_r) -> pd.DataFrame:

    # Save accuracy results
    results_test_score = pd.DataFrame(columns=['Model', 'Month', 'Type', 'MSE', 'MAE', 'VAR', 'R2', 'MAPE'])
    results_train_score = pd.DataFrame(columns=['Model', 'Month', 'Type', 'MSE', 'MAE', 'VAR', 'R2', 'MAPE'])

    # Models used for loop
    model_number = 15
    models = [
        ('LinR', LinearRegression(n_jobs=-1)),
        ('Ridge', Ridge(random_state=random_seed)),
        ('Lasso', Lasso(alpha=0.1)),
        ('Ada', AdaBoostRegressor(n_estimators=1000, learning_rate=0.0001, loss='linear', random_state=1)),
        ('GBR', GradientBoostingRegressor(learning_rate=0.0001, n_estimators=1000,
                                          max_depth=10, random_state=1, max_features=10)),
        ('Bag', BaggingRegressor(n_estimators=50, random_state=1)),
        ('RF', RandomForestRegressor(n_estimators=1000, max_depth=20)),
        ('Bayes', BayesianRidge(n_iter=60000)),
        ('EN',  ElasticNet(max_iter=60000, random_state=1)),
        ('Kernel', KernelRidge(alpha=1.0,  kernel='linear',  degree=10)),
        ('DT', DecisionTreeRegressor()),
        ('XGBR', XGBRegressor(booster='gblinear', objective="reg:squarederror",
                              random_state=1, n_estimators=1000, max_depth=10, learning_rate=0.0001)),
        ('CBR', CatBoostRegressor()),
        ('LGBR', LGBMRegressor()),
    ]
    level0 = models.copy()
    stack_model = StackingRegressor(estimators=level0,
                                    final_estimator=RandomForestRegressor(n_estimators=1000, random_state=random_seed))
    models.append(('Stack', stack_model))

    # if len(train_ind_r) < 3 or len(test_ind_r) < 3:
    for name, model in models:
        # save predicted results (training(fire months) & testing(fire and non-fire months)) for each model
        column_list = ['Type', 'raw_index', 'Month', 'Modeled', 'Observed', 'Error_Adjusted']
        train_result_df, test_result1_df, test_result0_df = \
            [pd.DataFrame(columns= column_list,
                          data=np.zeros(length, 6)) for length in [train_ind_r, test_ind_r, test_ind_0_r]]

        # Only models have enough data could perform regression
        if len(train_ind_r) > 3 and len(test_ind_r) > 3:
            clf = model.fit(X_train_r, y_train_r)

            # Predict fire size with the selected predicted fire records and non-fire records for each month
            y_train_fitted = clf.predict(X_train_r)
            y_pred_r = clf.predict(X_test_r)

            # Read classification error
            FDR, FOR, NPV, PPV = read_classification_error(path_error)

            # save fitting scores
            scores = metrics(y_test_r, y_pred_r)
            scores2 = [name, month, 'testing'] + scores
            results_test_score.loc[len(results_test_score.index)] = scores2
            scores_train = metrics(y_train_r, y_train_fitted)
            scores_train2 = [name, month, 'training'] + scores_train
            results_train_score.loc[len(results_train_score.index)] = scores_train2
            modeled_result = [y_train_fitted, y_pred_r, 0]
            error_adj_result = [y_train_fitted, y_pred_r * PPV + FDR * 0, y_test_0_r * FOR + NPV * 0]
        else:
            results_test_score.loc[len(results_test_score.index)] = [name, month, 'testing', np.nan, np.nan, np.nan,
                                                                     np.nan, np.nan]
            results_train_score.loc[len(results_train_score.index)] = [name, month, 'training', np.nan, np.nan, np.nan,
                                                                       np.nan, np.nan]
            modeled_result = [0, 0, 0]
            error_adj_result = [0, 0, 0]

        # Save results
        train_result_df.loc[:, 0], test_result1_df.loc[: 0], test_result0_df.loc[: 0] = [1, 1, 0]
        train_result_df.loc[:, 1], test_result1_df.loc[: 1], test_result0_df.loc[: 1] = [train_ind_r, test_ind_r, test_ind_0_r]
        train_result_df.loc[:, 2], test_result1_df.loc[: 2], test_result0_df.loc[: 2] = [month, month, month]
        train_result_df.loc[:, 3], test_result1_df.loc[: 3], test_result0_df.loc[: 3] = modeled_result
        train_result_df.loc[:, 4], test_result1_df.loc[: 4], test_result0_df.loc[: 4] = [y_train_r.values, y_test_r.values, y_test_0_r.values]
        train_result_df.loc[:, 5], test_result1_df.loc[: 5], test_result0_df.loc[: 5] = error_adj_result
        results_train_array = train_result_df
        results_test_array = np.concatenate((test_result1_df, test_result0_df), axis=0)

        # Write results to excel
        results_train_array.to_excel(writer, sheet_name=str(month), index=True)
        results_test_array.to_excel(writer_all, sheet_name=str(month), index=True)

    return writer, writer_all, results_test_score, results_train_score, results_train_array, results_test_array


def read_iputs(path_input):
    # file = path_input + 'clean_data_sea.xlsx'
    file = path_input + 'classification_input.xlsx'

    train_ind = pd.read_excel(file, sheet_name='ind_train', index_col=0)
    test_ind = pd.read_excel(file, sheet_name='ind_test', index_col=0)

    # original training and testing data for regression
    df_train_reg = pd.read_excel(file, sheet_name='training_std_before_oversampling', index_col=0)
    df_train_reg.set_index('raw_index')
    df_test_reg = pd.read_excel(file, sheet_name='testing_std', index_col=0)
    df_test_reg.set_index('raw_index')

    return train_ind, test_ind, df_train_reg, df_test_reg


def main():
    # Define input path and create output paths
    path_input, input_location, path_error, \
    path_output_train, path_output_test = create_directories('file_paths_regression.txt')

    # Define result dataframe to save accuracies
    accuracy_TST = pd.DataFrame(columns=['Model', 'Month', 'Type', 'MSE', 'MAE', 'VAR', 'R2', 'MAPE'])
    accuracy_Train = pd.DataFrame(columns=['Model', 'Month', 'Type', 'MSE', 'MAE', 'VAR', 'R2', 'MAPE'])

    # Write results to excel
    file_out_prd = path_output_train + 'Regression_' + name + '_train_prd.xlsx'
    file_out_prd_all = path_output_test + 'Regression_' + name + '_test_prd.xlsx'
    writer = pd.ExcelWriter(file_out_prd, engine='openpyxl')
    writer_all = pd.ExcelWriter(file_out_prd_all, engine='openpyxl')

    # Read inputs
    train_ind, test_ind, df_test_reg, df_train_reg = read_iputs(path_input)
    input_var = df_train_reg.columns[1:]
    target_var = df_test_reg.columns[-1]
    # input_var = ['row', 'column', 'Month', 'pet', 'tmn', 'cld', 'dtr', 'vap', 'wet', 'pre', 'tmp',
    #              'tmx', 'frs', 'RH', 'SVP', 'VPD', 'windspeed', 'ndvi',
    #              'PDSI', 'SMroot', 'SMsurf', 'BA_km2']  # 2 + 18 + 1
    # Classifiers = ['LogReg', 'RF', 'BAG', 'KNN', 'SVM', 'GNB', 'MLP', 'Volt']

    seasons = [('all-year', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])]


    for month in month_list:

        ind_selected_train = df_train_reg[(df_train_reg[target_var] > 0) & (df_train_reg['Month'] == month)].raw_index.values.tolist()
        ind_selected_test = df_test_pred[(df_test_pred['RF'] == 1) & (df_test_pred['Month'] == month)].raw_index.values.tolist()

        # monthly test data with fire happens
        X_test = df_test_reg[df_test_reg.raw_index.isin(ind_selected_test)].iloc[:, :-1]
        y_test = df_test_reg[df_test_reg.raw_index.isin(ind_selected_test)].iloc[:, -1]
        X_train = df_train_reg[df_train_reg.raw_index.isin(ind_selected_train)].iloc[:, :-1]
        y_train = df_train_reg[df_train_reg.raw_index.isin(ind_selected_train)].iloc[:, -1]

        ind_selected_train_0 = df_train_reg[
            (df_train_reg[input_var[-1]] == 0) & (df_train_reg['Month'] == month)
            ].raw_index.values.tolist()

        X_train_0 = df_train_reg[df_train_reg.raw_index.isin(ind_selected_train_0)].iloc[:, :-1]
        y_train_0 = df_train_reg[df_train_reg.raw_index.isin(ind_selected_train_0)].iloc[:, -1]

        ind_selected_test_0 = df_test_pred[
            (df_test_pred['RF'] == 0) & (df_test_pred['Month'] == month)
        ].raw_index.values.tolist()
        X_test_0 = df_test_reg[df_test_reg.raw_index.isin(ind_selected_test_0)].iloc[:, :-1]
        y_test_0 = df_test_reg[df_test_reg.raw_index.isin(ind_selected_test_0)].iloc[:, -1]

        print(X_train.shape, y_train.shape,
              X_train_0.shape, y_train_0.shape,
              X_test.shape, y_test.shape,
              X_test_0.shape, y_test_0.shape)

        accuracy_test, accuracy_train = run_exps_seasons(month,
                                                         path_error,
                                                         writer, writer_all,
                                                         X_train, y_train,
                                                         X_test, y_test,
                                                         ind_selected_train, ind_selected_test,
                                                         X_test_0, y_test_0, ind_selected_test_0)
        accuracy_TST = accuracy_TST.append(accuracy_test)
        accuracy_Train = accuracy_Train.append(accuracy_train)


    # save results df
    writer_all.save()
    writer_all.close()
    writer.save()
    writer.close()

    # Write results df
    file_out_metrics = path_output_test + 'regression_accuracy_summary.xlsx'
    writer = pd.ExcelWriter(file_out_metrics, engine='openpyxl')
    accuracy_TST.to_excel(writer, sheet_name='regression_accuracy_test_only', index=True)
    accuracy_Train.to_excel(writer, sheet_name='regression_accuracy_train', index=True)
    writer.save()
    writer.close()

if __name__ == "__main__":
    main()
