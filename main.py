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

delinquency = data_dict['delinquency']
historical = data_dict['historical'].drop(['date_from'],axis=1)
scenarios = data_dict['scenarios'].drop(['date_from'],axis=1)

plots.plotDelinquencyRates(delinquency)

endog = importData.transformData(delinquency, 'diff4')
plots.plotQoQDelinquencyRates(endog)


exog = importData.transformData(historical[historical.columns[5:]], 'diff1')
exog.columns = historical.columns[5:]

endog, exog, iidx = importData.matchIdx(endog, exog, lag=1)



#aDF, granger_results = statTests.testNonStationarity(endogQoQ, covariates=exogQoQ)

# # %% Fit vector autoregressive (VAR) model and check out of sample performance
# varData = varM.trainTestData(endog, exog, 12)


# results = {}

# empty = pd.DataFrame()
# results['VAR (1)'] = varM.varModel(varData, 1, 0, 1, X = 0)
# results['AR (1)'] = varM.varModel(varData, 1, 0, 0, X = 0)

# results['VARX (1)'] = varM.varModel(varData, 1, 0, 1, X = 1)
# results['ARX (1)'] = varM.varModel(varData, 1, 0, 0, X = 1)

# # importData.plotDF(forecasts, 'Delinquency (QoQ) Forecasts')
# # importData.plotDF(test_endog, 'Delinquency (QoQ)')


# # %% Fit the VAR Markov Switching model and check out of sample performance

# varMSData = varmsM.trainTestData(endog, exog, 1, 12)

# varmsHP = {}
# varmsHP['trend'] = 'n'
# varmsHP['var'] = True
# varmsHP['ar'] = True
# varmsHP['exog'] = True
# varmsHP['trend_switch'] = False


# empty = pd.DataFrame()
# #results['ARMS (1)'] = varmsM.varMSModel(varMSData, 1, 0, X = 0, hyperparams= varmsHP)

# results['ARXMS (1)'] = varmsM.varMSModel(varMSData, 1, 0, X = 1, hyperparams= varmsHP)

# # %% Create MSE matrix
# MSE = pd.DataFrame()
# colnames = []
# for key in results.keys():
#     MSE = pd.concat([MSE,(results[key])['MSE']],axis=1)
#     colnames.append(key)
# MSE.columns = colnames
# MSE = MSE.T

# # Divide by benchmark
# MSE_comparison = MSE/MSE.loc['ARX (1)']
# MSE_weighted = MSE/abs(varMSData['test_endog'].mean(axis=0))