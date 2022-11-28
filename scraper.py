import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from numpy import mean
import os
import datetime
import statsmodels.api as sm
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from pandas.plotting import register_matplotlib_converters
from sklearn.tree import DecisionTreeClassifier
from selenium import webdriver

circuits= pd.read_csv('circuits.csv')
weather = circuits.iloc[:,[0,1,2]]

info = []

# read wikipedia tables

for link in circuits.url:
    try:
        df = pd.read_html(link)[0]
        if 'Length' in list(df.iloc[:,0]):
            n = list(df.iloc[:,0]).index('Length')
            info.append(df.iloc[n,1])
        else:
            df = pd.read_html(link)[1]
            if 'Length' in list(df.iloc[:,0]):
                n = list(df.iloc[:,0]).index('Length')
                info.append(df.iloc[n,1])
            else:
                df = pd.read_html(link)[2]
                if 'Length' in list(df.iloc[:,0]):
                    n = list(df.iloc[:,0]).index('Length')
                    info.append(df.iloc[n,1])
                else:
                    df = pd.read_html(link)[3]
                    n = list(df.iloc[:,0]).index('Length')
                    info.append(df.iloc[n,1])

                                
    except:
        info.append('not found')
        
circuits['Length'] = info

circuits.loc[circuits["circuitRef"] =='long_beach', "Length"] = '3.275 km (2.035 mi)'
circuits.loc[circuits["circuitRef"] =='las_vegas', "Length"] = "6.116 km (3.800 mi)"
circuits.loc[circuits["circuitRef"] =='zeltweg', "Length"] = "3.186 km (1.980 mi)"
circuits.loc[circuits["circuitRef"] =='shanghai', "Length"] = "5.451 km (3.387 mi)"

circuits.to_csv('circuits.csv',index=False)
