# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 20:15:02 2023

@author: hp
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def data_reading(file):
    r_data = pd.read_csv(file, skiprows = 4)
    return r_data

def filter_data(df, col, value, con, yr):
    df1 = df.groupby(col, group_keys= True)
    df1 = df1.get_group(value)
    df1 = df1.reset_index()
    df1.set_index('Country Name', inplace=True)
    df1 = df1.loc[:, yr]
    df1 = df1.loc[con, :]
    df1= df1.dropna(axis = 1)
    df1 = df1.reset_index()
    df2 = df1.set_index('Country Name')  
    df2=df2.transpose()
    return df1,df2

dataset =  data_reading("Gender_Equality.csv")


countries_bar= ['China','Pakistan','India','Indonesia']
year_bar = ['1990', '1995','2000','2005','2010','2015','2020']
data_bar, data_bar_t = filter_data(dataset, 'Indicator Name','Unemployment, male (% of male labor force) (modeled ILO estimate)', countries_bar, year_bar)
print(data_bar)
print(data_bar_t)