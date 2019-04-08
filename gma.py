#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 14:52:09 2019

@author: saneelprabhu
"""
import pandas as pd
import numpy as np

#import matplotlib.pyplot as plt

#imports
costs = pd.read_csv('costs.csv').sort_values(by=['calendar_date','tracking_key'])
bookings = pd.read_csv('bookings.csv').sort_values(by=['calendar_date','tracking_key'])
revenue = pd.read_csv('revenue.csv').sort_values(by=['calendar_date','tracking_key'])

#datetimes
costs['calendar_date'] = pd.to_datetime(costs['calendar_date'], format='%Y-%m-%d')
bookings['calendar_date'] = pd.to_datetime(bookings['calendar_date'], format='%Y-%m-%d')
revenue['calendar_date'] = pd.to_datetime(revenue['calendar_date'], format='%Y-%m-%d')

#CATEGORICALS VARIABLES FUNCTION 
categoricals = costs.columns[2:11].tolist()

def category(column):
    costs[column] = costs[column].astype('category')
    
for i in categoricals:
    category(i)

#joining bookings and revenue tables 
br = pd.merge(bookings, revenue, how='left')

#full data set 
df = pd.merge(costs,br,how='inner')


#split df into df w/ dupes and df w/o dupes
df_dupes = df[df.duplicated(['calendar_date','tracking_key', 'bookings', 'revenue'],keep=False) == True]

df_dupes = df_dupes.groupby(['calendar_date','tracking_key'],as_index=False).agg({'impressions':sum,
                           'clicks':sum,
                           'cost':sum, 
                           'bookings':np.mean,
                           'revenue':np.mean,})

#remove dupes
df_no_dupes = df[df.duplicated(['calendar_date','tracking_key', 'bookings', 'revenue'],keep=False) == False]
df_no_dupes = df_no_dupes.groupby(['calendar_date' ,'tracking_key'],as_index=False).sum()

#Rejoin df_dupes and df_no_dupes
data = pd.concat([df_no_dupes,df_dupes], ignore_index=True)

#function to compute ad metrics for a given dataframe
#def get_stats(df):
#    #advertising metrics
#    df['cpm'] = df.cost * 1000 / df.impressions
#    df['click_rate'] = df.clicks / df.impressions
#    df['cpc'] = df.cost / df.clicks
#    df['cost_per_acq'] = df.cost / df.bookings
#    df['roas'] = (df.revenue - df.cost)/ df.cost
#
#    df['cpc'] = df['cpc'].replace(np.inf, np.nan)
#    df['cost_per_acq'] = df['cost_per_acq'].replace(np.inf, np.nan)


#dictionary of columns and agg funcs


aggy = {'impressions':'sum',
                          'clicks':'sum',
                          'cost':'sum', 
                          'bookings':'sum', 
                          'revenue':'sum'} 

agg_metrics = {'impressions':'sum',
                          'clicks':'sum',
                          'cost':'sum', 
                          'bookings':'sum', 
                          'revenue':'sum', 
                          'click_rate':'mean',
                          'cpc':'mean',
                          'cost_per_acq':'mean',
                          'roas':'mean'} 

weekly_data = data.groupby([pd.Grouper(key='calendar_date',freq='W')]).agg(aggy)
    
monthly_data = data.groupby([pd.Grouper(key='calendar_date',freq='MS')]).agg(aggy)
    
#GROUP BY PLATFORM/CHANNEL 

#split data
pc_no_dupes = df[df.duplicated(['calendar_date','tracking_key','platform','channel', 'bookings', 'revenue'],keep=False) == False]
pc_no_dupes = pc_no_dupes.groupby(['calendar_date','tracking_key', 'platform','channel'],as_index=False).agg({'impressions':sum,'clicks':sum,'cost':sum, 'bookings':sum,'revenue':sum})

#de-dupe
pc_dupes = df[df.duplicated(['calendar_date','tracking_key','platform','channel', 'bookings', 'revenue'],keep=False) == True]
pc_dupes = pc_dupes.groupby(['calendar_date','tracking_key', 'platform','channel'],as_index=False).agg({'impressions':sum,'clicks':sum,'cost':sum, 'bookings':np.mean,'revenue':np.mean})

#Platform/Channel Daily Data 
pc = pd.concat([pc_no_dupes,pc_dupes], ignore_index=True)

#aggregate platform/channel data by totals,weekly, and month
pc_data = pc.groupby(['channel','platform']).agg(aggy).sort_values(by='bookings', ascending=False).dropna(axis=0,how='any')


############## OVERALL CONVERSION STATS FOR TOP 3 CHANNELS — Table 1 #######################

def cpa(df):
    #df.loc[:,'cpc'] = df.loc[:,'cost'] / df.loc[:,'clicks']
    #df.loc[:,'conv. rate'] = df.loc[:,'bookings'] / df.loc[:,'clicks']
    df.loc[:,'cost_per_acq'] = df.loc[:,'cost'] / df.loc[:,'bookings']
    df.loc[:,'roas'] = (df.loc[:,'revenue'] - df.loc[:,'cost'])/ df.loc[:,'cost']
    
    
############## OVERALL CONVERSION STATS  — Table 1 #######################


pc_data.index = pc_data.index.map('/'.join).str.strip('/')

pc_data_cb = pc_data[['cost','bookings','revenue']]

pc_data_cb.loc[:,'% of costs'] = pc_data.loc[:,'cost'] / pc_data.loc[:,'cost'].sum()
pc_data_cb.loc[:,'% of bookings'] = pc_data.loc[:,'bookings'] / pc_data.loc[:,'bookings'].sum()
pc_data_cb.loc[:,'% of revenue'] = pc_data.loc[:,'revenue'] / pc_data.loc[:,'revenue'].sum()

pc_data_cb = pc_data_cb[['cost','% of costs','bookings','% of bookings','revenue','% of revenue']]



table1 = pc_data_cb.head(3)

#cpa(table1)
table1_final = table1.T

def get_percents(df):
    
    df.loc[:,'% of costs'] = df.loc[:,'cost'] / df.loc[:,'cost'].sum()
    df.loc[:,'% of bookings'] = df.loc[:,'bookings'] / df.loc[:,'bookings'].sum()
    df.loc[:,'% of revenue'] = df.loc[:,'revenue'] / df.loc[:,'revenue'].sum()
    

    return(df)

############## OVERALL CONVERSION STATS FOR SEM/GOOGLE SUB-CHANNELS — Table 3 #######################

sem_no_dupes = df[df.duplicated(['calendar_date','tracking_key','platform','channel','campaign_strategy','match_type','bookings', 'revenue'],keep=False) == False]
sem_no_dupes = sem_no_dupes.groupby(['calendar_date','tracking_key','platform','channel','campaign_strategy','match_type'],as_index=False).agg({'cost':sum,'clicks':sum ,'bookings':sum,'revenue':sum})

sem_dupes = df[df.duplicated(['calendar_date','tracking_key','platform','channel','campaign_strategy','match_type', 'bookings', 'revenue'],keep=False) == True]
sem_dupes = sem_dupes.groupby(['calendar_date','tracking_key', 'platform','channel','campaign_strategy','match_type'],as_index=False).agg({'cost':sum,'clicks':sum ,'bookings':'median','revenue':'min'})

sem = pd.concat([sem_no_dupes,sem_dupes], ignore_index=True)

subchannel_data = sem.groupby(['channel','platform','match_type', 'campaign_strategy'],as_index=False).agg(sum).sort_values(by='bookings', ascending=False).dropna(axis=0,how='any')
subchannel_data = subchannel_data.set_index(['channel','platform'])
subchannel_data.index = subchannel_data.index.map('/'.join).str.strip('/')

#SEM BRAND 
sembrand_sub = subchannel_data.loc['SEM Brand/google']
get_percents(sembrand_sub)
cpa(sembrand_sub)
sembrand_sub = sembrand_sub[['match_type','campaign_strategy','cost','% of costs','bookings','% of bookings','cost_per_acq','roas']]


##SEM NON-BRAND 
sem_nonbrand_sub = subchannel_data.loc['SEM Non-brand/google']
get_percents(sem_nonbrand_sub)

sem_nonbrand_sub = sem_nonbrand_sub[['match_type','campaign_strategy','cost','% of costs','bookings','% of bookings','revenue']]
cpa(sem_nonbrand_sub)
#sem_nonbrand_sub = sem_nonbrand_sub.T
#
############### OVERALL CONVERSION STATS FOR SOCIAL/FACEBOOK SUB-CHANNELS — Table 4 #######################

social = df[(df['platform'] =='facebook') & (df['channel'] == 'Social')]
#
social_no_dupes = social[social.duplicated(['calendar_date','tracking_key','platform','channel','campaign_strategy','is_remarketing','bookings', 'revenue'],keep=False) == False]
social_no_dupes = social_no_dupes.groupby(['calendar_date','tracking_key','platform','channel','campaign_strategy','is_remarketing'],as_index=False).agg({'cost':sum, 'bookings':sum,'revenue':sum})
#
social_dupes = social[social.duplicated(['calendar_date','tracking_key','platform','channel','campaign_strategy','is_remarketing','bookings', 'revenue'],keep=False) == True]
social_dupes = social_dupes.groupby(['calendar_date','tracking_key', 'platform','channel','campaign_strategy','is_remarketing'],as_index=False).agg({'cost':sum, 'bookings':'median','revenue':'min'})
#
social_fb = pd.concat([social_no_dupes,social_dupes], ignore_index=True)
#
social_subchannel_data = social.groupby(['channel','platform','is_remarketing', 'campaign_strategy'],as_index=False).agg(sum).sort_values(by='bookings', ascending=False).dropna(axis=0,how='any')
social_subchannel_data = social_subchannel_data.set_index(['channel','platform'])
social_subchannel_data.index = social_subchannel_data.index.map('/'.join).str.strip('/')
#
##social_sub = subchannel_data.loc['SEM Brand/google']
get_percents(social_subchannel_data)
social_subchannel_data = social_subchannel_data[['campaign_strategy','is_remarketing','cost','% of costs','bookings','% of bookings','revenue']]
cpa(social_subchannel_data)

#
############### GEOGRAPHIC BREAKDOWNS — Table 7 #######################
#
#language = costs.groupby('language').agg(sum).sort_values(by='clicks', ascending=False)
#origin_country = costs.groupby('dim_origin_country').agg(sum).sort_values(by='clicks', ascending=False)
#origin_region = costs.groupby('dim_origin_region').agg(sum).sort_values(by='clicks', ascending=False)
#destination = costs.groupby('destination_country').agg(sum).sort_values(by='clicks', ascending=False)
#
#
############### TOP/BOTTOM 10 CAMPAIGNS — Table 8 #######################
#
#top10_campaigns = df.groupby('tracking_key').agg(sum).sort_values(by='bookings',ascending=False).head(10)
#
#
############### MONTHLY/WEEKLY — Table 2 #######################
#
#pc_by_month = pc.groupby([pd.Grouper(key='calendar_date',freq='MS'), 'channel', 'platform']).agg(aggy).dropna(axis=0,how='any').reset_index()
#pc_by_week = pc.groupby([pd.Grouper(key='calendar_date',freq='W'), 'channel', 'platform']).agg(aggy).dropna(axis=0,how='any').reset_index()
#    
#pc_by_month = pc_by_month.set_index(['channel','platform'])
#pc_by_month.index = pc_by_month.index.map('/'.join).str.strip('/')
#pc_by_month = pc_by_month.loc[['SEM Brand/google','SEM Non-brand/google','Social/facebook']]
#pc_by_month = pc_by_month[['calendar_date','cost','bookings','revenue']]
#cpa(pc_by_month)
#pc_by_month = pc_by_month.sort_values(by='calendar_date')
#pc_by_month = pc_by_month.T
#
#pc_by_week = pc_by_week.set_index(['channel','platform'])
#pc_by_week.index = pc_by_week.index.map('/'.join).str.strip('/')
#pc_by_week = pc_by_week.loc[['SEM Brand/google','SEM Non-brand/google','Social/facebook']]
#pc_by_week = pc_by_week[['calendar_date','cost','bookings','revenue']]
#cpa(pc_by_week)
#pc_by_week = pc_by_week.sort_values(by='calendar_date')
#pc_by_week = pc_by_week.T
#
#
############### MONTHLY CONVERSION STATS FOR SOCIAL/FACEBOOK SUB-CHANNELS — Table 4 #######################
#
#social_monthly_sub_data = social_fb.groupby([pd.Grouper(key='calendar_date',freq='MS'),'channel','platform','is_remarketing', 'campaign_strategy'],as_index=False).agg(sum).sort_values(by='bookings', ascending=False).dropna(axis=0,how='any')
#
#
#
