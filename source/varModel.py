#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 20:44:35 2022

@author: tgarutti
"""

# %% Import packages
import statsmodels.api as sm
import numpy as np
import pandas as pd

# %% Vector autoregressive model
def varModel(varData, ar, ma, X):
    train_endog = varData['train_endog']
    test_endog = varData['test_endog'] 
    
    if X==0:
        train_exog = test_exog = pd.DataFrame()
    else:
        train_exog = varData['train_exog']
        test_exog = varData['test_exog']
        
    
    n_forecasts = len(test_endog)
    startDate = test_endog.index[0]
    mod, modelFit = varModelFit(train_endog, ar, ma, train_exog)
    forecasts = varModelForecast(modelFit, n_forecasts, test_exog, startDate)
    residuals = modelFit.resid
    
    estimationErrors = test_endog-forecasts
    MSE = np.power(estimationErrors, 2).mean(axis=0)
    
    results={}
    results['forecasts'] = forecasts
    results['estimationErrors'] = estimationErrors
    results['residuals'] = residuals
    results['MSE'] = MSE
    
    return results

def varModelFit(endog, ar, ma, exog):
    if exog.empty:
        mod = sm.tsa.VARMAX(endog, order=(ar,ma), trend='n')
    else:
        mod = sm.tsa.VARMAX(endog, order=(ar,ma), trend='n', exog=exog)
    modelFit = mod.fit(maxiter=1000, disp=False)
    
    return mod, modelFit

def varModelForecast(modelFit, n_forecasts, exog, startDate):
    if exog.empty:
        forecast = modelFit.simulate(nsimulations=n_forecasts, anchor = startDate)
    else:
        forecast = modelFit.simulate(nsimulations=n_forecasts, exog=exog, anchor = startDate)
    return forecast

# %% Process data for VAR model
def trainTestData(endog, exog, n_test):
    data = {}
    
    endog_sorted = endog.sort_index(ascending = True)
    data["test_endog"] = endog_sorted[-n_test:]
    data["train_endog"] = endog_sorted[:-n_test]
    
    exog_sorted = exog.sort_index(ascending = True)
    data["test_exog"] = exog_sorted[-n_test:]
    data["train_exog"] = exog_sorted[:-n_test]
    return data