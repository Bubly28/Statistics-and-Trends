# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 20:15:02 2023

@author: hp
"""

import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns


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


def stat_fn(s_df, col_s, val_s, yr_s, i_s):
    '''
    This function used to filter data for heatmap and statistical tools

    Parameters
    ----------
    s_df : Dataset
    col_s : Name of the required column
    val_s : Value in the chosen column
    yr_s : Years for filtering data
    i_s : Indicators for filtering data

    Returns
    -------

    df_stat : Filtered dataset

    '''

    # Grouping the dataset by a column name
    df_stat = s_df.groupby(col_s, group_keys=True)
    # Taking the dataset of a particular column value
    df_stat = df_stat.get_group(val_s)
    # Resetting the index of the grouped dataset
    df_stat = df_stat.reset_index()
    # Setting Indicator Name as the index
    df_stat.set_index('Indicator Name', inplace=True)
    # Cropping the data by years
    df_stat = df_stat.loc[:, yr_s]
    # Transposing the filtered dataset
    df_stat = df_stat.transpose()
    # Cropping the data by indicators
    df_stat = df_stat.loc[:, i_s]
    # Dropping the nan values
    df_stat = df_stat.dropna(axis=1)
    # Return statement is used at the end of a function
    return df_stat


def bar_plot(f_data_b, title_name_b, x_lab_b, y_lab_b):
    '''
    This function is used for bar plotting

    Parameters
    ----------
    f_data : The filtered dataset for bar plotting
    title_name : Title of the bar plot
    x_lab : x-label
    y_lab : y-label

    '''

    bx = f_data_b.plot.bar(x='Country Name', rot=0,
                           figsize=(50, 30), fontsize=50)
    # Setting yticks
    bx.set_yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    # Setting the title for the plot
    bx.set_title(title_name_b.upper(), fontsize=60, fontweight='bold')
    # Setting the x-label for the plot
    bx.set_xlabel(x_lab_b, fontsize=50)
    # Setting the y-label for the plot
    bx.set_ylabel(y_lab_b, fontsize=50)
    bx.legend(fontsize=50)
    plt.savefig(title_name_b + '.png')
    plt.show()
    # Return statement is used at the end of a function
    return


def line_plot(f_data_l, title_name_l, x_lab_l, y_lab_l):
    '''
    This function is used for line plotting

    Parameters
    ----------
    f_data_l : The filtered dataset for line plotting
    title_name_l : Title of the line plot
    x_lab_l : x-label
    y_lab_l : y-label

    '''

    lx = f_data_l.plot.line(figsize=(50, 30), fontsize=36, linewidth=7.0)
    # Setting yticks
    lx.set_yticks([50, 55, 60, 65, 70, 75, 80, 85, 90])
    # Setting the title for the plot
    lx.set_title(title_name_l.upper(), fontsize=60, fontweight='bold')
    # Setting the x-label for the plot
    lx.set_xlabel(x_lab_l, fontsize=50)
    # Setting the y-label for the plot
    lx.set_ylabel(y_lab_l, fontsize=50)
    lx.legend(fontsize=50)
    plt.savefig(title_name_l + '.png')
    plt.show()
    # Return statement is used at the end of a function
    return


def heat_map(data_h, con_h, gen, color):
    '''
    This function is used to plot heatmap

    Parameters
    ----------
    data_h : dataset filtered for plotting heatmap
    con_h : Country name
    gen : Gender 
    color : Color of the heatmap

    Returns
    -------
    data_h : The dataset used for plotting heatmap

    '''
    plt.figure(figsize=(20, 18))
    h_map = sns.heatmap(data_h.corr(), annot=True, cmap=color)
    # Setting title for heatmap
    h_map.set_title(con_h + '(' + gen + ')')
    plt.savefig(con_h + gen + ".png", dpi=300, bbox_inches='tight')
    # Return statement is used at the end of a function
    return data_h


# Calling the function for reading data and assigning it in a variable
dataset = data_reading("Gender_Equality.csv")


# List of countries for bar plotting
countries_bar = ['Indonesia', 'Pakistan', 'India', 'China', 'Mongolia']
# List of years for bar plotting
year_bar = ['1994', '1998', '2002', '2006', '2010', '2014', '2018']
# Calling the data filtering function for 1st bar plot and assinging its values in variables
data_bar1, data_bar1_t = filter_data(
    dataset, 'Indicator Name', 'Unemployment, male (% of male labor force) (modeled ILO estimate)', countries_bar, year_bar)
# Printing the dataset with years as columns
print(data_bar1)
# Printing the transposed dataset with countries as columns
print(data_bar1_t)
# Calling the function for 1st bar plotting
bar_plot(data_bar1, 'Unemployment (male)',
         'Countries', 'Unemployment rate')

# Calling the data filtering function for 2nd bar plot and assinging its values in variables
data_bar2, data_bar2_t = filter_data(
    dataset, 'Indicator Name', 'Unemployment, female (% of female labor force) (modeled ILO estimate)', countries_bar, year_bar)
# Printing the dataset with years as columns
print(data_bar2)
# Printing the transposed dataset with countries as columns
print(data_bar2_t)
# Calling the function for 2nd bar plotting
bar_plot(data_bar2, 'Unemployment (female)',
         'Countries', 'Unemployment rate')


# List of countries for line plotting
countries_line = ['Japan', 'Iraq', 'Kuwait', 'Afghanistan', 'Philippines']
# List of years for line plotting
year_line = ['1995', '1998', '2001', '2004',
             '2007', '2010', '2013', '2016', '2019']
# Calling the data filtering function for 1st line plot and assinging its values in variables
data_line1, data_line1_t = filter_data(
    dataset, 'Indicator Name', 'Life expectancy at birth, male (years)', countries_line, year_line)
# Printing the dataset with years as columns
print(data_line1)
# Printing the transposed dataset with countries as columns
print(data_line1_t)
# Calling the function for 2nd line plotting
line_plot(data_line1_t, 'Life expectancy at birth (male)',
          'Year', 'Life expectancy rate')


# Calling the data filtering function for 2nd line plot and assinging its values in variables
data_line2, data_line2_t = filter_data(
    dataset, 'Indicator Name', 'Life expectancy at birth, female (years)', countries_line, year_line)
# Printing the dataset with years as columns
print(data_line2)
# Printing the transposed dataset with countries as columns
print(data_line2_t)
# Calling the function for 2nd line plotting
line_plot(data_line2_t, 'Life expectancy at birth (female)',
          'Year', 'Life expectancy at birth rate')

# List of years for plotting heatmap
years_h = ['2000', '2002', '2004', '2006', '2008', '2010']

# List of indicators for plotting 1st heatmap
ind_h_m = ['Mortality rate, adult, male (per 1,000 male adults)', 'Survival to age 65, male (% of cohort)', 'Unemployment, male (% of male labor force) (modeled ILO estimate)', 'School enrollment, tertiary, male (% gross)',
           'Employment in services, male (% of male employment) (modeled ILO estimate)', 'Wage and salaried workers, male (% of male employment) (modeled ILO estimate)', 'Self-employed, male (% of male employment) (modeled ILO estimate)']
# Calling the stat function
d_heat = stat_fn(dataset, 'Country Name', 'India', years_h, ind_h_m)
# Calling the function for plotting the 1st heatmap
heat_map(d_heat, 'India', 'Male', 'RdPu')

# List of indicators for plotting 2nd heatmap
ind_h_f = ['Mortality rate, adult, female (per 1,000 female adults)', 'Survival to age 65, female (% of cohort)', 'Unemployment, female (% of female labor force) (modeled ILO estimate)', 'School enrollment, tertiary, female (% gross)',
           'Employment in services, female (% of female employment) (modeled ILO estimate)', 'Wage and salaried workers, female (% of female employment) (modeled ILO estimate)', 'Self-employed, female (% of female employment) (modeled ILO estimate)']
# Calling the stat function
d_heat = stat_fn(dataset, 'Country Name', 'India', years_h, ind_h_f)
# Calling the function for plotting the 2nd heatmap
heat_map(d_heat, 'India', 'Female', 'GnBu')

start = 1995
end = 2015
year_des = [str(i) for i in range(start, end+1)]
ind_des = ['Unemployment, male (% of male labor force) (modeled ILO estimate)', 'Unemployment, female (% of female labor force) (modeled ILO estimate)',
           'Life expectancy at birth, male (years)', 'Life expectancy at birth, female (years)']
data_des = stat_fn(dataset, 'Country Name', 'China', year_des, ind_des)
# Describing the data
summary_statistics = data_des.describe()
# Finding skewness
skewness = stats.skew(data_des['Life expectancy at birth, male (years)'])
# Finding kurtosis
kurtosis = data_des['Life expectancy at birth, female (years)'].kurtosis()
# Printing the value of skewness
print('Skewness of Life expectancy of male in China : ', skewness)
# Printinf the value of kurtosis
print('Kurtosis of Life expectancy of female in China : ', kurtosis)
# Saving the file in csv format
summary_statistics.to_csv("China's Summary Statistics.csv")
