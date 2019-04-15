import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing
from team_targets import *

def get_clean_pdata():
    allplayers = pd.read_csv('data/pg_pstats_05-19.csv')

    allplayers = allplayers[allplayers.Tm != 'TOT']
    allplayers = allplayers.drop_duplicates(subset=['Player','YR'], keep='first')

    #allplayers.drop(['Unnamed: 19', 'Unnamed: 24'], axis=1, inplace=True)

    tconv = pd.read_csv('data/Pstats - teams.csv')

    apf = pd.merge(allplayers, tconv, on='Tm', how='outer')

    apf['TM'] = apf['Team'] + ' ' + apf['YR']

    apf.drop(['Rk', 'Tm','YR', 'Team'], axis=1, inplace=True)

    apf = apf[apf['Player'] != 'Player']

    apf = apf.dropna()

    apf = apf[['Player', 'TM', 'G','MP', 'PTS']]
    apf['PTS'] = apf['MP'].astype('float')
    apf['MP'] = apf['MP'].astype('float')
    apf['G'] = apf['G'].astype('int')

    ap_10 = apf

    #remove asterisk from player name
    ap_10['Player'] = ap_10['Player'].str.replace('*', '')

    ap_10['YR'] = ap_10['TM'].apply(lambda x : x[-2:]).astype('int32')

    ap_10['YRprior'] = ap_10['YR'] - 1


    return ap_10

def add_yr_prior(ap_10):
    aprior = ap_10

    ap = pd.merge(ap_10, aprior, how='outer', left_on=['Player','YRprior'], right_on = ['Player','YR'])

    return ap

#combine main player data with d_votes table
pdata = get_clean_pdata()


pdata = pdata[pdata['G'] > 16]

#Create new df of player data with prioer year
p2 = add_yr_prior(pdata)

#Drop rows that don't have a team assigned to it
p3 = p2.dropna(subset=['TM_x'])

# #Fill NaNs with that position/team data so that it aggregates correctly
p3 = p3.fillna(p3.groupby('TM_x').transform('mean'))

# #Aggregate players' data into team data by players who played an average of most min. over last 2 years

p310 = p3.sort_values('MP_y', ascending=False).groupby('TM_x').head(10)

p10 = p310.groupby('TM_x').mean().reset_index()
p10fill = p10.fillna(p10.groupby('TM_x').quantile(.2))

#Get target of teams making top 4 of conference
team_target = get_team_target()

#Merge aggregated players (not in teams) with targets to lineup training with y
team_df = pd.merge(p10fill, team_target, how='left', left_on='TM_x', right_on="TM_x")
team_df_fill0 = team_df.fillna(team_df.quantile(.2))

#drop year 6, since it is beginning of data and has no prior data
team_df_fin = team_df_fill0[team_df_fill0.YR_x !=5]

#drop NaNs
team_fin = team_df_fin.dropna().reset_index()

#Create csv of final team data
team_fin.to_csv('data/5_19_pergame.csv', index=False)


print('done loading')