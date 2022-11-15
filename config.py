#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 21:15:42 2022

@author: tgarutti
"""

# %% Define paths
import os
path = os.path.abspath(os.getcwd())

input_path = path+'/01_Input/'
output_path = path+'/02_Output/'

# %% Input file name and worksheet names

file_name = 'raw_data.xlsx'

sheets = {'chargeoff' : 'Charge-Off Rates NSA all banks',  
         'delinquency' : 'Delinquency Rates NSA all banks', 
         'historical' : 'Historical economic data', 
         'scenarios' : 'Economic scenario'}