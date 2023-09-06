import numpy as np
import pandas as pd

ad_clicks = pd.read_csv('dataset/ad_clicks.csv')
total_users = ad_clicks.user_id.count()
#print(total_users)
#print(ad_clicks.head())
viewed_platforms = ad_clicks.groupby('utm_source').user_id.count().reset_index()
viewed_platforms.rename(columns={
    'utm_source': 'source',
    'user_id': 'views'
}, inplace='True')
#print(viewed_platforms)
ad_clicks['is_click'] = ~ad_clicks.ad_click_timestamp.isnull()
#print(ad_clicks.head())

clicks_by_source = ad_clicks.groupby(['utm_source', 'is_click']).user_id.count().reset_index()
#print(clicks_by_source)
clicks_pivot = clicks_by_source.pivot(columns='is_click', index='utm_source', values='user_id').reset_index()
#print(clicks_pivot)
clicks_pivot['percent_clicked'] = (clicks_pivot[True]/(clicks_pivot[True]+clicks_pivot[False]))*100
#print(clicks_pivot)

ad_stat = ad_clicks.groupby('experimental_group').user_id.count().reset_index()
#print(ad_stat)
#print(ad_clicks.head())
is_click = ad_clicks.groupby(['experimental_group', 'is_click']).user_id.count().reset_index()
#print(is_click)
a_clicks = ad_clicks[ad_clicks.experimental_group == 'A'].reset_index(drop=True)
#print(a_clicks.head())
b_clicks = ad_clicks[ad_clicks.experimental_group == 'B'].reset_index(drop=True)
#print(b_clicks.head())
ad_click_b_by_day = b_clicks.groupby(['day', 'is_click']).user_id.count().reset_index()
#print("Stat on ad B clicks:\n", ad_click_b_by_day)
ad_click_a_by_day = a_clicks.groupby(['day', 'is_click']).user_id.count().reset_index()
print("Stat on ad A clicks:\n", ad_click_a_by_day)
ad_click_b_by_day_pivot = ad_click_b_by_day.pivot(columns='is_click', values='user_id', index='day').reset_index()
ad_click_a_by_day_pivot = ad_click_a_by_day.pivot(columns='is_click', values='user_id', index='day').reset_index()

ad_click_b_by_day_pivot['Percentage clicks'] = 100*(ad_click_b_by_day_pivot[True]/(ad_click_b_by_day_pivot[True]+ad_click_b_by_day_pivot[False]))
ad_click_a_by_day_pivot['Percentage clicks'] = 100*(ad_click_a_by_day_pivot[True]/(ad_click_a_by_day_pivot[True]+ad_click_a_by_day_pivot[False]))
print("TABLE on ad B clicks:\n", ad_click_b_by_day_pivot)
print("TABLE on ad A clicks:\n", ad_click_a_by_day_pivot)
ad_click_b_by_day_pivot_mean = ad_click_b_by_day_pivot['Percentage clicks'].mean()
ad_click_a_by_day_pivot_mean = ad_click_a_by_day_pivot['Percentage clicks'].mean()
print("Mean % clicks over the week for ad B:-\n", ad_click_b_by_day_pivot_mean)
print("Mean % clicks over the week for ad A:-\n", ad_click_a_by_day_pivot_mean)
