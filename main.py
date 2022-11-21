#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 14:30:37 2022

@author: tgarutti
"""

# %% Import packages
import numpy as np
import pandas as pd
from datetime import datetime

import source.importData as importData
import source.statisticalTests as statTests
import source.varModel as varM
import source.varmsModel as varmsM
import source.plots as plots

# %% Read and process raw data
data_raw, data_dict = importData.readData()
#plots.plotAllData(data_dict)

delinquency = data_dict['delinquency'].sort_index(ascending = True)
historical = data_dict['historical'].drop(['date_from'],axis=1)
scenarios = data_dict['scenarios'].drop(['date_from'],axis=1)

endog = importData.transformData(delinquency, 'diff4')
exog = importData.transformData(historical[scenarios.columns], 'sort')
exog.columns = historical.columns[5:]

endog, exog, iidx = importData.matchIdx(endog, exog, lag=1)

# Plots
# plots.plotDelinquencyRates(delinquency)
# plots.plotQoQDelinquencyRates(endog)


# Summary Statistics

#aDF_org, _ = statTests.testNonStationarity(delinquency, covariates=exog)
#aDF, _ = statTests.testNonStationarity(endog, covariates=exog)

# %% Fit vector autoregressive (VAR) model and check out of sample performance
varData = varM.trainTestData(endog, exog, 12)


results = {}

empty = pd.DataFrame()
results['VAR (1)'] = varM.varModel(varData, 1, 0, 1, X = 0)
results['AR (1)'] = varM.varModel(varData, 1, 0, 0, X = 0)

results['VARX (1)'] = varM.varModel(varData, 1, 0, 1, X = 1)
results['ARX (1)'] = varM.varModel(varData, 1, 0, 0, X = 1)

# importData.plotDF(forecasts, 'Delinquency (QoQ) Forecasts')
# importData.plotDF(test_endog, 'Delinquency (QoQ)')


# %% Fit the VAR Markov Switching model and check out of sample performance

varMSData = varmsM.trainTestData(endog, exog, 1, 12)

varmsHP = {}
varmsHP['trend'] = 'n'
varmsHP['var'] = True
varmsHP['ar'] = False
varmsHP['exog'] = True
varmsHP['trend_switch'] = False


empty = pd.DataFrame()
#results['ARMS (1)'] = varmsM.varMSModel(varMSData, 1, 0, X = 0, hyperparams= varmsHP)

#results['ARXMS (1)'] = varmsM.varMSModel(varMSData, 1, 0, X = 1, hyperparams= varmsHP)

# %% Create MSE matrix
MSE_QoQ = pd.DataFrame()
colnames = []
for key in results.keys():
    MSE_QoQ = pd.concat([MSE_QoQ,(results[key])['MSE']],axis=1)
    colnames.append(key)
MSE_QoQ.columns = colnames
MSE_QoQ = MSE_QoQ.T

# Divide by benchmark
MSE_QoQ_comparison = MSE_QoQ/MSE_QoQ.loc['ARX (1)']
MSE_QoQ_weighted = MSE_QoQ/abs(varMSData['test_endog'].mean(axis=0))


# %% Reverse transformation
endog_train = endog[:-12]
for key in results.keys():
    forecastsQoQ = results[key]['forecastsQoQ']
    results[key]['forecasts'] = importData.reverseTransformedData(delinquency.loc[endog_train.index], forecastsQoQ, "diff4")
    results[key]['FullSampleResults'] = pd.concat([delinquency.loc[endog_train.index], results[key]['forecasts']])
    plots.plotForecasts(results[key]['FullSampleResults'], 12, key)
    
# %% Create MSE matrix
test_del = delinquency.loc[endog[-12:].index]
MSE = pd.DataFrame()
colnames = []
for key in results.keys():
    mse = np.power((test_del - results[key]['forecasts']),2).mean(axis=0)
    MSE = pd.concat([MSE,mse],axis=1)
    colnames.append(key)
MSE.columns = colnames
MSE = MSE.T

# Divide by benchmark
MSE_comparison = MSE/MSE.loc['ARX (1)']
MSE_weighted = MSE/abs(varMSData['test_endog'].mean(axis=0))

# %% Perform scenario forecast
scenarios.columns = historical.columns[5:]
data_scenario = importData.getScenarioData(endog, exog, scenarios)

results_scenario = varmsM.varMSModel(data_scenario, 1, 0, X = 1, hyperparams= varmsHP)
forecastsMS = importData.reverseTransformedData(delinquency.loc[endog.index], results_scenario['forecastsQoQ'], "diff4")
plots.plotForecasts(pd.concat([delinquency.loc[endog.index],forecastsMS]), 7, "Forecasted Delinquencies (MS(X) model)")

residsMS = results_scenario['residuals'].dropna().std(axis=0)
