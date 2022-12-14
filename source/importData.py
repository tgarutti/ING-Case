#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 21:15:14 2022

@author: xhoxha
"""

# %% Packages

import pandas as pd
import numpy as np
from copy import deepcopy 
import matplotlib.pyplot as plt

# %% Read raw data
def readData():
    from config import input_path, file_name, sheets

    file_path = input_path + file_name
    
    # Read charge-off data
    data_chargeoff = pd.read_excel(file_path, sheet_name=sheets['chargeoff'],
                                   skiprows=3, header=None, index_col=0)
    
    # Read delinquency data
    data_delinquency = pd.read_excel(file_path, sheet_name=sheets['delinquency'],
                                     skiprows=3, header=None, index_col=0)
    
    col_names = ['all_real_estate', 'residential', 'commercial', 'farmland', 
                 'all_consumer', 'credit_card', 'other', 
                 'leases', 'C&I', 'agricultural', 'total'] 
   
    data_chargeoff.columns = data_delinquency.columns = col_names
    
    # Remove unnecessary columns
    ignore_cols = {'residential', 'commercial', 'farmland', 
                   'credit_card', 'other', 'total'}
    select_cols = [col for col in col_names if col not in ignore_cols]
    data_chargeoff = data_chargeoff[select_cols]
    data_delinquency = data_delinquency[select_cols]
    
    # Replace 'n.a.' values
    data_chargeoff.replace('n.a.', np.nan, inplace=True)
    data_delinquency.replace('n.a.', np.nan, inplace=True)
    
    # Read historical data
    data_historical = pd.read_excel(file_path, sheet_name=sheets['historical'],
                                    skiprows=3, header=None, index_col=1)
    
    col_names = ['date_from', '10y_swap', '3m_interbank', 'GDP', 
                 'HPI', 'unemployment', 'QoQ_10y_swap', 
                 'QoQ_3m_interbank', 'QoQ_GDP', 'QoQ_HPI', 'QoQ_unemployment']
    
    data_historical.columns = col_names
    
    # Read economic scenario data
    data_scenarios = pd.read_excel(file_path, sheet_name=sheets['scenarios'],
                                   skiprows=3, header=None, index_col=1)
    
    col_names = ['date_from', '10y_swap', '3m_interbank', 'unemployment',
                 'QoQ_GDP', 'QoQ_HPI']
    
    data_scenarios.columns = col_names
    
    data = {'chargeoff': data_chargeoff, 
            'delinquency': data_delinquency, 
            'historical': data_historical, 
            'scenarios': data_scenarios}
    
    # Save an instance of the raw data
    data_raw = deepcopy(data)

    # Remove empty values
    for name in data.keys():
        data[name].dropna(axis=0, how='any', inplace=True)
        
    return data_raw, data
        
# %% Process data functions
def transformData(data, transformation):
    data = data.sort_index(ascending = True)
    if transformation == 'log_diff1':
        data = np.log(data).diff(periods=1).iloc[1:]
    if transformation == 'diff1':
        data = data.diff(periods=1).iloc[1:]
    if transformation == 'diff4':
        data = data.diff(periods=4).iloc[4:]
    if transformation == 'diff1_diff4':
        data = data.diff(periods=1).iloc[1:]
        data = data.diff(periods=4).iloc[4:]
    if transformation == 'diff4_diff1':
            data = data.diff(periods=4).iloc[4:]
            data = data.diff(periods=1).iloc[1:]
    return data


def reverseTransformedData(orig_data, forecasted_data, transformation):
    data = pd.concat([orig_data, forecasted_data])

    for i in range(len(orig_data), len(data)):
        if transformation == 'log_diff1':
            data.iloc[i] = data.iloc[i] + data.iloc[i-1]
            data.iloc[i] = np.exp(data.iloc[i])
        if transformation == 'diff1':
            data.iloc[i] = data.iloc[i] + data.iloc[i-1]
        if transformation == 'diff4':
            data.iloc[i] = data.iloc[i] + data.iloc[i-4]
        if transformation == 'diff1_diff4':
            data.iloc[i] = data.iloc[i] + data.iloc[i-4]
            data.iloc[i] = data.iloc[i] + data.iloc[i-1]
        if transformation == 'diff4_diff1':
            data.iloc[i] = data.iloc[i] + data.iloc[i-1]
            data.iloc[i] = data.iloc[i] + data.iloc[i-4]

    orig_forecasted_data = data[-len(forecasted_data):]
    return orig_forecasted_data


def matchIdx(endog, exog, lag=1):
    idx_endog = endog.index
    idx_exog = exog.index.shift(lag, freq='Q')
    exog.index = idx_exog
    iidx = idx_endog.intersection(idx_exog)
    return endog.loc[iidx], exog.loc[iidx], iidx
       
def getScenarioData(endog, exog, scenario):
    train_endog = endog
    train_exog = exog
    test_endog = scenario*0
    test_endog.columns = train_endog.columns
    test_exog = scenario
    
    data = {}
    
    data["test_endog"] = test_endog
    data["train_endog"] = train_endog
    
    data["test_exog"] = test_exog
    data["train_exog"] = train_exog
    
    return data