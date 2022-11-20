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
    plt.figure()
    train_del = delinquency[:-n_forecasts]
    forecasts = delinquency[-n_forecasts:]
    my_colors = list(islice(cycle(['b','y','g','r','m']), None, len(train_del)))
    plt.plot(train_del, linestyle= '-', color = my_colors)
    plt.plot(forecasts, linestyle= '--', color = my_colors)

    for i, col in enumerate(train_del.columns):
        plt.plot(train_del[col], "-", colors[i])
        plt.plot(forecasts[col], "--", colors[i])
    
    plt.ylabel('delinquency rate')
    plt.legend(['All real estate', 'All customer', 'Leases', 'C&I', 'Agricultural'])
    

    #usrec = DataReader('USREC', 'fred', start=datetime(1987, 1, 1), end=datetime(2022, 4, 1))
    #plt.fill_between(usrec.index, 0, 1, where=usrec['USREC'].values, color='k', alpha=0.1)
    plt.ylim([0,0.12])


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