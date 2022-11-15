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
historical = data_dict['historical']
scenarios = data_dict['scenarios']



aDF, delinquencyDiff, aDF_firstDiff, delinquencyLog, aDF_log, za_results = \
    statTests.testNonStationarity(delinquency, diff=4)


    
train_y, test_y = importData.trainTestData(delinquencyLog, 12)
    
# %% 