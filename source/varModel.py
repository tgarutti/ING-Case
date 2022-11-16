#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 20:44:35 2022

@author: tgarutti
"""

# %% Import packages
import statsmodels.api as sm
import numpy as np

# %% Vector autoregressive model
def varModel(train_endog, ar, ma, train_exog, test_endog, test_exog):
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
    modelfit = mod.fit(maxiter=1000, disp=False)
    
    return mod, modelfit

def varModelForecast(modelFit, n_forecasts, exog, startDate):
    if exog.empty:
        forecast = modelFit.simulate(nsimulations=n_forecasts, anchor = startDate)
    else:
        forecast = modelFit.simulate(nsimulations=n_forecasts, exog=exog, anchor = startDate)
    return forecast