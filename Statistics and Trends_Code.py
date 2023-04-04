# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 20:15:02 2023

@author: hp
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def data_reading(file):
    '''
    This function is used to read the dataset in csv format.
    The name of the csv file is passed as an argument.
    '''

    # read the csv file
    r_data = pd.read_csv(file, skiprows=4)
    # Return statement is used at the end of a function
    return r_data


def filter_data(data_r, col_name,  col_val, contries, years):
    '''
    This function is used to filter the dataset.

    Parameters
    ----------
    data_r : The dataset which read using the function data_reading()
    col_name : The column name
    col_val : The value in the chosen column
    contries : Countries for filtering the data
    years : Years for filtering the data

    Returns
    -------
    data_filter : Dataset with years as columns
    data_filter_trans :Dataset with countries as columns

    '''
    # Grouping the dataset by a column name
    data_filter = data_r.groupby(col_name, group_keys=True)
    # Taking the dataset of a particular column value
    data_filter = data_filter.get_group(col_val)
    # Resetting the index of the grouped dataset
    data_filter = data_filter.reset_index()
    # Setting Country Name as the index
    data_filter.set_index('Country Name', inplace=True)
    # Cropping the data by years
    data_filter = data_filter.loc[:, years]
    # Cropping the data by countries
    data_filter = data_filter.loc[contries, :]
    # Dropping the nan values
    data_filter = data_filter.dropna(axis=1)
    # Resetting the index
    data_filter = data_filter.reset_index()
    # Setting Country Name as the index
    data_filter_trans = data_filter.set_index('Country Name')
    # Transposing the filtered dataset
    data_filter_trans = data_filter_trans.transpose()
    # Return statement is used at the end of a function
    return data_filter, data_filter_trans


def bar_plot(f_data, title_name, x_lab, y_lab):
    '''
    This function is used for bar plotting

    Parameters
    ----------
    f_data : The filtered dataset for plotting
    title_name : The title of the bar plot
    x_lab : x-label
    y_lab : y-label

    '''

    bx = f_data.plot.bar(x='Country Name', rot=0,
                         figsize=(50, 30), fontsize=50)
    # Setting yticks
    bx.set_yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    # Setting the title for the plot
    bx.set_title(title_name.upper(), fontsize=60, fontweight='bold')
    # Setting the x-label for the plot
    bx.set_xlabel(x_lab, fontsize=50)
    # Setting the y-label for the plot
    bx.set_ylabel(y_lab, fontsize=50)
    bx.legend(fontsize=50)
    plt.savefig(title_name + '.png')
    plt.show()
    # Return statement is used at the end of a function
    return


# Calling the function for reading data and assigning it in a variable
dataset = data_reading("Gender_Equality.csv")


# List of countries for bar plotting
countries_bar = ['Indonesia', 'Pakistan', 'India', 'China', 'Mongolia']
# List of years for bar plotting
year_bar = ['1998', '2002', '2006', '2010', '2014', '2016', '2020']
# Calling the data filtering function for 1st bar plot and assinging its values in variables
data_bar1, data_bar1_t = filter_data(
    dataset, 'Indicator Name', 'Unemployment, male (% of male labor force) (modeled ILO estimate)', countries_bar, year_bar)
# Printing the dataset with years as columns
print(data_bar1)
# Printing the transposed dataset with countries as columns
print(data_bar1_t)
# Calling the function for 1st bar plotting
bar_plot(data_bar1, 'Unemployment (male)',
         'Countries', 'Unemployment rate (males, % of male labour force)')

# Calling the data filtering function for 2nd bar plot and assinging its values in variables
data_bar2, data_bar2_t = filter_data(
    dataset, 'Indicator Name', 'Unemployment, female (% of female labor force) (modeled ILO estimate)', countries_bar, year_bar)
# Printing the dataset with years as columns
print(data_bar2)
# Printing the transposed dataset with countries as columns
print(data_bar2_t)
# Calling the function for 2nd bar plotting
bar_plot(data_bar2, 'Unemployment (female)',
         'Countries', 'Unemployment rate (females, % of female labor force)')
