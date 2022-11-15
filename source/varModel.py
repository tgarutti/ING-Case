#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 20:44:35 2022

@author: tgarutti
"""

# %% Import packages
import statsmodels.api as sm

# %% Vector autoregressive model
def varModel(endog, ar, ma, exog):
    mod = sm.tsa.VARMAX(endog, order=(ar,ma), trend='n', exog=exog)
    modelfit = mod.fit(maxiter=1000, disp=False)
    
    return mod, modelfit