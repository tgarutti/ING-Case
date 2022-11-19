#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 14:30:37 2022

@author: tgarutti
"""

# %% Import packages
import source.importData as importData
import source.statisticalTests as statTests
import numpy as np
import pandas as pd
from datetime import datetime
import source.varModel as varM
import source.varmsModel as varmsM

# %% Read and process raw data
data_raw, data_dict = importData.readData()
#importData.plotAllData(data_dict)

delinquency = data_dict['delinquency']
historical = data_dict['historical'].drop(['date_from'],axis=1)
scenarios = data_dict['scenarios'].drop(['date_from'],axis=1)

endog = importData.transformData(delinquency, 'diff4')
exog = importData.transformData(historical[historical.columns[5:]], 'sort')

endog, exog, iidx = importData.matchIdx(endog, exog, lag=1)

#aDF, granger_results = statTests.testNonStationarity(endogQoQ, covariates=exogQoQ)

# %% Fit vector autoregressive (VAR) model and check out of sample performance
varData = varM.trainTestData(endog, exog, 12)


results = {}

empty = pd.DataFrame()
results['VAR (1)'] = varM.varModel(varData, 1, 0, X = 0)

results['VARX (1)'] = varM.varModel(varData, 1, 0, X = 1)

# importData.plotDF(forecasts, 'Delinquency (QoQ) Forecasts')
# importData.plotDF(test_endog, 'Delinquency (QoQ)')


# %% Fit the VAR Markov Switching model and check out of sample performance

varMSData = varmsM.trainTestData(endog, exog, 1, 12)

empty = pd.DataFrame()
#results['VARMS (1)'] = varmsM.varMSModel(varMSData, 1, 0, X = 0)

results['VARXMS (1)'] = varmsM.varMSModel(varMSData, 2, 0, X = 1)