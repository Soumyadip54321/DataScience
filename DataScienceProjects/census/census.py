import numpy as np
import pandas as pd
import regex as re


def stamp_highertax(x):
    if 'strongly disagree' == x:
        return -2
    elif 'disagree' == x:
        return -1
    elif 'neutral' == x:
        return 0
    elif 'agree' == x:
        return 1
    else:
        return 2


def result(num):
    if num == 0:
        return 'neutral'
    elif num == -1:
        return 'disagree'
    elif num == -2:
        return 'strongly disagree'
    elif num == 1:
        return 'agree'
    else:
        return 'strongly agree'



census_df = pd.read_csv('census.csv')
#print(census_df.head())
#print(census_df.info())
census_df.rename(columns={"Unnamed: 0": "serial_num"}, inplace=True)
#print(census_df.head())
#print(census_df.dtypes)
#print(census_df.birth_year.unique())
census_df.birth_year = census_df['birth_year'].replace('missing', '0')
#print(census_df.birth_year.unique())
census_df.birth_year = census_df.birth_year.astype("int64")
#print(census_df.dtypes)
#print("average birth year of respondents\n", int(census_df.birth_year.mean()))
census_df.higher_tax = pd.Categorical(census_df.higher_tax, ['strongly disagree', 'disagree', 'neutral', 'agree', 'strongly agree'], ordered=True)
#print(census_df.higher_tax.unique())
census_df['higher_tax_label'] = census_df.apply(lambda x: stamp_highertax(x.higher_tax), axis=1)
#print(census_df.head())
#print(census_df.dtypes)
median_sentiment = census_df.higher_tax_label.median()
print("The median sentiment is\n", result(median_sentiment))
census_df = pd.get_dummies(data=census_df, columns=['marital_status'])
print(census_df.head())
census_df = census_df.drop('higher_tax_label', axis=1)
print(census_df.head())