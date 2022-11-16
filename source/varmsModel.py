#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 14:40:04 2022

@author: tgarutti
"""

# %% Import packages
import statsmodels.api as sm
import numpy as np

# %% Vector autoregressive model
def varModel(train_endog, ar, ma, train_exog, test_endog, test_exog):
    n_forecasts = len(test_endog)
    startDate = test_endog.index[0]
    
    
    estimationErrors = test_endog-forecasts
    MSE = np.power(estimationErrors, 2).mean(axis=0)
    
    return forecasts, residuals, MSE

def varModelFit(endog, ar, ma, exog):
    mod = sm.tsa.VARMAX(endog, order=(ar,ma), trend='n', exog=exog)
    modelfit = mod.fit(maxiter=1000, disp=False)
    
    return mod, modelfit

def varModelForecast(modelFit, n_forecasts, exog, startDate):
    forecast = modelFit.simulate(nsimulations=n_forecasts, exog=exog, anchor = startDate)
    return forecast