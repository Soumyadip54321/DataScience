import pandas as pd
import numpy as np

data = pd.read_csv('page_visits.csv')
#print(data.head())
unique_sources = data.utm_source.unique()
#print(unique_sources)
click_source = data.groupby('utm_source').id.count().reset_index()
#print(click_source)
click_source_by_month = data.groupby(['utm_source', 'month']).id.count().reset_index()
#print(click_source_by_month)
click_source_by_month_pivot = click_source_by_month.pivot(columns='month', index='utm_source', values='id')#.reset_index()
print(click_source_by_month_pivot)
