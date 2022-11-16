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
import source.varModel as arModels

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
train_endog, test_endog = importData.trainTestData(endog, 12)
train_exog, test_exog = importData.trainTestData(exog, 12)

results = {}

empty = pd.DataFrame()
results['VAR (1)'] = arModels.varModel(train_endog, 1, 0, empty, test_endog, empty)

results['VARX (1)'] = arModels.varModel(train_endog, 1, 0, train_exog, test_endog, test_exog)

# importData.plotDF(forecasts, 'Delinquency (QoQ) Forecasts')
# importData.plotDF(test_endog, 'Delinquency (QoQ)')


# %% Fit the VAR Markov Switching model and check out of sample performance