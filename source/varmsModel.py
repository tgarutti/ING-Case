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

# %% Vector autoregressive model
def varMSModel(varMSData, ar, ma, X):
    train_endog = varMSData['train_endog']
    test_endog = varMSData['test_endog'] 
    
    if X==0:
        train_exog = test_exog = pd.DataFrame()
    else:
        train_exog = varMSData['train_exog']
        test_exog = varMSData['test_exog']
        
    n_forecasts = len(test_endog)
    startDate = test_endog.index[0]
    forecasts = test_endog
    residuals = test_endog
    
    for col in train_endog.columns:
        endog = train_endog[col]
        exog = train_exog
        if X == 1:
            col_exog = []
            for i in range(ar):
                lag=i+1
                col_exog = col_exog + [col+"_Lag("+str(lag)+")"]
            col_exog = col_exog+train_exog.columns[5:].to_list()
            exog = train_exog[col_exog]
        mod, modelFit = varMSModelFit(endog, ar, ma, exog)
        forecasts[col] = varMSModelForecast(modelFit, n_forecasts, test_exog[col_exog], startDate)
        #residuals[col] = modelFit.resid
        
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
        mod = sm.tsa.MarkovRegression(endog, k_regimes=2, 
              trend='c', exog=exog, switching_variance=True)
    modelFit = mod.fit(maxiter=1000, disp=False)
    
    return mod, modelFit

def varMSModelForecast(modelFit, n_forecasts, exog, startDate):
    forecast = modelFit.predict(exog = exog)
    
    # if exog.empty:
    #     forecast = modelFit.simulate(nsimulations=n_forecasts, anchor = startDate)
    # else:
    #     forecast = modelFit.simulate(nsimulations=n_forecasts, exog=exog, anchor = startDate)
    return forecast


# %% Process data for VAR MS model
def trainTestData(endog, exog, ar, n_test):
    for i in range(ar):
        lag = i+1
        endog_ar = endog[:-(lag)]
        idx = endog_ar.index.shift(lag, freq='Q')
        endog_ar.index = idx
        cols = endog.columns
        cols = [col + "_Lag("+str(lag)+")" for col in cols]
        endog_ar.columns = cols
        exog_ar = pd.concat([endog_ar, exog.loc[idx]], axis=1)
        
    endog = endog.loc[idx]
    
    data = {}
    
    endog_sorted = endog.sort_index(ascending = True)
    data["test_endog"] = endog_sorted[-n_test:]
    data["train_endog"] = endog_sorted[:-n_test]
    
    exog_sorted = exog_ar.sort_index(ascending = True)
    data["test_exog"] = exog_sorted[-n_test:]
    data["train_exog"] = exog_sorted[:-n_test]
    return data