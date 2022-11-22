#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 18:22:25 2022

@author: tgarutti
"""
# %% Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, zivot_andrews, grangercausalitytests
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm

# %% Testing Non-Stationaritys
def testNonStationarity(data, covariates):
    # 1) Augmented Dickey-Fuller test
    aDF = augmentedDickeyFuller(data)
    #za_results = zivotAndrews(data)
    
    #plotACF_PCAF(data)
    #dec = decompose(data,'additive')
        
    granger_results = grangerCausality(data, covariates, maxlags=4)

    return aDF, granger_results


# %% Non-Stationarity Tests
def augmentedDickeyFuller(data):
    colnames = data.columns if type(data) == pd.DataFrame else data.columns
    
    aDF = pd.DataFrame()
    for col in colnames:
        data_col = data[col]
        result = adfuller(data_col)
        aDF_temp = pd.DataFrame([result[0], result[1]], 
                           index = ['ADF Statistics', 'p-value'],
                           columns = [col])
        aDF = pd.concat([aDF,aDF_temp],axis=1)
        
    return aDF


def zivotAndrews(data, regression='c'):
    colnames = data.columns if type(data) == pd.DataFrame else data.names
    
    za_results = pd.DataFrame()
    for col in colnames:
        data_col = data[col]
        result = zivot_andrews(data_col,regression = regression, trim=0.01)
        za_temp = pd.DataFrame([result[0], result[1]], 
                           index = ['ZA Statistics', 'p-value'],
                           columns = [col])
        za_results = pd.concat([za_results,za_temp],axis=1)
        
    return za_results


def grangerCausality(data, covariates, maxlags = 4):
    colnames = data.columns if type(data) == pd.DataFrame else data.names
    covnames = covariates.columns if type(covariates) == pd.DataFrame else covariates.names
    
    granger_results = {}
    for col in colnames:
        data_col = data[col]
        for cov in covnames:
            print('Endogenous: ' + col)
            print('Exogenous: ' + cov)
            print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -')
            data_cov = covariates[cov]
            try:
                granger = grangercausalitytests(pd.concat([data_col.reset_index(drop=True), data_cov.reset_index(drop=True)],axis=1), maxlag = maxlags)
                granger_pvalue = [granger[i][0]['ssr_ftest'][1] for i in range(1,maxlags)]
                granger_results[col + ' & ' + cov] = granger_pvalue
            except ValueError:
                print('ERROR: Granger Causality test failed.')
                pass
            print('--------------------------------------------------------------------------')
            
    return granger_results


def plotACF_PCAF(data):
    colnames = data.columns if type(data) == pd.DataFrame else data.names
    
    for col in colnames:
        data_col = data[col]
        fig = plt.figure(figsize=(12,8))
        ax1 = fig.add_subplot(211)
        fig = plot_acf(data_col.values.squeeze(), lags=40, ax=ax1)
        ax2 = fig.add_subplot(212)
        fig = plot_pacf(data_col, lags=40, ax=ax2)
            

def decompose(data, model='additive'):
    colnames = data.columns if type(data) == pd.DataFrame else data.columns
    
    for col in colnames:
        data_col = data[col]
        result = seasonal_decompose(data_col, model='additive')
        result.plot()
        plt.show()
    return result

def regression(data):
    agricultural = data['agricultural']
    N = len(agricultural)
    dummy_Q1 = [1 if agricultural.index[i].month==3 else 0 for i in range(N)]
    dummy_Q2 = [1 if agricultural.index[i].month==6 else 0 for i in range(N)]
    dummy_Q3 = [1 if agricultural.index[i].month==9 else 0 for i in range(N)]
    
    Y = agricultural
    t = list(range(1,N+1))
    X = pd.DataFrame([t, dummy_Q1, dummy_Q2, dummy_Q3], 
                     index=['t', 'Q1', 'Q2', 'Q3'],
                     columns=agricultural.index).T
    model = sm.OLS(Y,X)
    results = model.fit()
    results.params
    print(results.t_test([1, 0]))

#%% Residuals tests

def normalQQ(resid):
    plt.rc('figure', figsize=(10,10))
    plt.style.use('ggplot')
    
    probplot = sm.ProbPlot(resid, fit=True)
    fig = probplot.qqplot(line='45', marker='o', color='black')
    plt.title('Normal Q-Q', fontsize=20)
    plt.show()