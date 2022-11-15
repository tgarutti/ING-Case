#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 14:30:37 2022

@author: tgarutti
"""

# %% Import packages
import source.importData as importData
import source.statisticalTests as statTests

# %% Read and process raw data
data_raw, data_dict = importData.readData()
importData.plotAllData(data_dict)

delinquency = data_dict['delinquency']
historical = data_dict['historical'].drop(['date_from'],axis=1)
scenarios = data_dict['scenarios'].drop(['date_from'],axis=1)



aDF, delinquencyDiff, aDF_firstDiff, delinquencyLog, aDF_log, za_results = \
    statTests.testNonStationarity(delinquency, diff=4)

iidx = historical.index.intersection(delinquencyLog.index)
endog = delinquencyLog.loc[iidx].sort_index(ascending = False)
exog = historical.loc[iidx].sort_index(ascending = False)[scenarios.columns]
    
train_endog, test_endog = importData.trainTestData(endog, 12)
train_exog, test_exog = importData.trainTestData(exog, 12)


# %% Train models