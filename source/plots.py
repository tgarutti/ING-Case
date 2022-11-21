import matplotlib.pyplot as plt

# NBER recessions
from pandas_datareader.data import DataReader
from datetime import datetime
from itertools import cycle, islice


# %% Plot data
def plotDelinquencyRates(delinquency):
    plt.figure()
    plt.plot(delinquency)
    plt.ylabel('delinquency rate')
    plt.legend(['All real estate', 'All customer', 'Leases', 'C&I', 'Agricultural'])

    usrec = DataReader('USREC', 'fred', start=datetime(1987, 1, 1), end=datetime(2022, 4, 1))
    #plt.title('Delinquency Rates (NSA) from 1987 to 2022 for all commercial banks')
    plt.fill_between(usrec.index, 0, 1, where=usrec['USREC'].values, color='k', alpha=0.1)
    plt.ylim([0,0.12])


def plotQoQDelinquencyRates(delinquency_QoQ):
    plt.figure()
    plt.plot(delinquency_QoQ)
    plt.ylabel('QoQ delinquency rate')
    plt.legend(['All real estate', 'All customer', 'Leases', 'C&I', 'Agricultural'])

    usrec = DataReader('USREC', 'fred', start=datetime(1987, 1, 1), end=datetime(2022, 4, 1))
    #plt.title('Delinquency Rates (NSA) from 1987 to 2022 for all commercial banks')
    plt.fill_between(usrec.index, -1, 1, where=usrec['USREC'].values, color='k', alpha=0.1)
    plt.ylim([-0.04,0.05])
    
    
def plotForecasts(delinquency, n_forecasts, model_name):
    fig, ax = plt.subplots()
    
    train_del = delinquency[-(n_forecasts+13):-n_forecasts+1]
    forecasts = delinquency[-n_forecasts:]
    
    # Get default colors
    prop_cycle = plt.rcParams['axes.prop_cycle']
    my_colors = prop_cycle.by_key()['color']
    
    train_del.columns = ['All real estate', 'All customer', 'Leases', 'C&I', 'Agricultural']
    train_del.plot(ax=ax, linestyle= '-', color = my_colors)
    forecasts.plot(ax=ax, linestyle= '--', color = my_colors, legend=False)
    ax.set_title(model_name)
    ax.set_ylim(0,0.04)
    ax.set_xlim()
    
def plotSmoothedProbabilities(smoothedP):
    endog_names = ['All real estate', 'All customer', 'Leases', 'C&I', 'Agricultural']
    fig, axs = plt.subplots(len(endog_names), figsize=(30,20))
    usrec = DataReader('USREC', 'fred', start=datetime(1987, 1, 1), end=datetime(2022, 4, 1))
    
    axs = axs.ravel()

    for i, col in enumerate(smoothedP.columns):
        row = smoothedP[col]
        axs[i].plot(row)
        axs[i].title.set_text("Smoothed Probabilities of Recession State (" + endog_names[i] + ")")
        axs[i].fill_between(usrec.index, -1, 1, where=usrec['USREC'].values, color='k', alpha=0.1)
        axs[i].set_ylim([0,1])
        axs[i].set_xlim([row.index[0],row.index[-1]])


def plotAllData(data):
    plotData(data, 'chargeoff')
    plotData(data, 'delinquency')
    plotHistoricalData(data['historical'])
    plotScenarioData(data['scenarios'])


def plotData(data, data_name):
    sub_data = data[data_name]
    col_names = sub_data.columns
    ignore_cols = {'residential', 'commercial', 'farmland', 
                   'credit_card', 'other', 'total'}
    col_list = [col for col in col_names if col not in ignore_cols]
    plt.figure()
    plt.plot(sub_data[col_list])
    plt.legend(col_list)  
    plt.title(data_name)
        

def plotHistoricalData(sub_data):
    col_names = sub_data.columns
    fig, axs = plt.subplots(5,2, figsize=(20, 20))
    k = 0
    for j in range(2):
        for i in range(5):
            axs[i,j].plot(sub_data[col_names[k+1]])
            axs[i,j].title.set_text(col_names[k+1])
            k = k + 1
    
def plotScenarioData(sub_data):
    col_names = sub_data.columns
    fig, axs = plt.subplots(5, figsize=(10,20))
    for i in range(5):
        axs[i].plot(sub_data[col_names[i+1]])
        axs[i].title.set_text(col_names[i+1])
        
def plotDF_subplots(df):
    col_names = df.columns
    fig, axs = plt.subplots(len(col_names), figsize=(30,20))
    for i in range(len(col_names)):
        axs[i].plot(df[col_names[i]])
        axs[i].title.set_text(col_names[i])
        
def plotDF(df, str_title):
    col_names = df.columns
    plt.figure()
    plt.plot(df)
    plt.legend(col_names)  
    plt.title(str_title)