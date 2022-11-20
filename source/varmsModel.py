#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 18:06:02 2022

@author: tgarutti
"""

# %% Import packages
import statsmodels.api as sm
import numpy as np
import pandas as pd
from copy import deepcopy

# %% Vector autoregressive model
def varMSModel(varMSData, ar, ma, X):
    train_endog = varMSData['train_endog']
    test_endog = varMSData['test_endog'] 
    
    if X==0:
        train_exog = test_exog = pd.DataFrame()
    else:
        train_exog = varMSData['train_exog']
        test_exog = varMSData['test_exog']
        
    full_endog = pd.concat([train_endog, test_endog])
    full_endog[-len(test_endog):] = 0
    
    forecasts = deepcopy(test_endog)
    residuals = deepcopy(train_endog)
    forecasts[:] = 0
    residuals[:] = 0
    
    for col in train_endog.columns:
        endog = train_endog[col]
        exog = train_exog
        mod, modelFit = varMSModelFit(endog, ar, ma, exog)
        forecasts[col] = varMSModelForecast(modelFit, ar, 2, test_exog, full_endog[col])
        residuals[col] = endog-modelFit.predict()
        
    # mod, modelFit = varMSModelFit(train_endog, ar, ma, train_exog)
    # forecasts = varMSModelForecast(modelFit, n_forecasts, test_exog, startDate)
    # residuals = modelFit.resid
    
    estimationErrors = test_endog-forecasts
    MSE = np.power(estimationErrors, 2).mean(axis=0)
    
    results={}
    results['forecasts'] = forecasts
    results['estimationErrors'] = estimationErrors
    results['residuals'] = residuals
    results['MSE'] = MSE
    
    return results

def varMSModelFit(endog, ar, ma, exog):
    if exog.empty:
        mod = sm.tsa.MarkovAutoregression(endog, order=ar, k_regimes=2, 
              trend='c', switching_variance=True)
    else:
        mod = sm.tsa.MarkovAutoregression(endog, order=ar, k_regimes=2, 
              trend='n', exog=exog, switching_variance=True,
              switching_exog=True)
    
    modelFit = mod.fit(maxiter=1000, disp=False)
    
    return mod, modelFit

def varMSModelForecast(modelFit, ar, k_regimes, exog, endog):
    n_forecasts = len(exog)
    p0 = modelFit.smoothed_marginal_probabilities[0][-1]
    p1 = modelFit.smoothed_marginal_probabilities[1][-1]
    
    params = processParams(modelFit.params, k_regimes)
    i = 0
    for idx, row in exog.iterrows():
        p0, p1 = calcProbabilities(p0, p1, params["Transition Probabilities"])
        
        forecasted_regimes = []
        for k in range(k_regimes):
            str_reg = "Regime "+str(k)
            ar_vec = endog[(-(n_forecasts+ar)+i):(-n_forecasts+i)]
            f_regime = forecastRegime(params[str_reg], row, ar_vec)
            forecasted_regimes.append(f_regime)
        forecast = p0*forecasted_regimes[0] + p1*forecasted_regimes[1]
        endog[idx] = forecast
    
    return endog[-n_forecasts:]


# %% Process data for VAR MS model
def trainTestData(endog, exog, ar, n_test):
    data = {}
    
    endog_sorted = endog.sort_index(ascending = True)
    data["test_endog"] = endog_sorted[-n_test:]
    data["train_endog"] = endog_sorted[:-n_test]
    
    exog_sorted = exog.sort_index(ascending = True)
    data["test_exog"] = exog_sorted[-n_test:]
    data["train_exog"] = exog_sorted[:-n_test]
    return data

# %% Helper functions
def processParams(input_mat, k_regimes):
    params = {}
    
    p00 = input_mat[0]
    p10 = input_mat[1]
    params['Transition Probabilities'] = np.array([[p00, 1-p00],[p10, 1-p10]])
    
    input_params = input_mat[k_regimes:]
    for i in range(k_regimes):
        str_regime = "["+str(i)+"]"
        name_regime = "Regime "+str(i)
        reg_params = input_params[[True if x > 0 else False for x in input_params.index.str.find(str_regime)]]
        reg_params.index = [x[:-3] for x in reg_params.index]
        if reg_params.index.str.contains('sigma').any():
            sigma_param = reg_params[reg_params.index.str.contains('sigma')]
        else:
            sigma_param = input_params[input_params.index.str.contains('sigma')]
        ar_params = reg_params[reg_params.index.str.contains('ar')]
        exog_params = reg_params[:-(len(sigma_param) + len(ar_params))]
        params_reg = {}
        params_reg["Exog Params"] = exog_params
        params_reg["AR Params"] = ar_params
        params_reg["Sigma"] = sigma_param
        params[name_regime] = params_reg
        
    return params


def calcProbabilities(p0, p1, trans_prob):

    # P[s(t+1) = 0 | s(t) = 0]
    p00 = trans_prob[0][0]
    # P[s(t+1) = 0 | s(t) = 1]
    p10 = trans_prob[1][0]
    
    pi0 = p00 * p0 + p10 * p1

    pi1 = 1 - pi0
    return pi0, pi1

def forecastRegime(params_reg, exog, endog):
    exog_params = params_reg["Exog Params"]
    ar_params = params_reg["AR Params"]
    ar_params.index = endog.index
    sigma = params_reg["Sigma"]
    
    
    f_exog = exog_params*exog
    f_ar = ar_params*endog
    e = np.random.normal(0, sigma, 1)
    
    forecast = sum(f_exog) + sum(f_ar) + e
    
    return forecast