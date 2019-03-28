import pandas as pd
import numpy as np

allplayers = pd.read_csv('data/adv_stats_06-18.csv')

allplayers = allplayers[allplayers.Tm != 'TOT']

allplayers.drop(['Unnamed: 19', 'Unnamed: 24'], axis=1, inplace=True)

tconv = pd.read_csv('data/Pstats - teams.csv')

apf = pd.merge(allplayers, tconv, on='Tm', how='outer')

apf['TM'] = apf['Team'] + ' ' + apf['YR']

apf.drop(['Rk', 'Tm', 'Team'], axis=1, inplace=True)

apf = apf[apf['Player'] != 'Player']

apf = apf.dropna()

apf['OBPM'] = apf['OBPM'].astype('float')
apf['DBPM'] = apf['DBPM'].astype('float')
apf['BPM'] = apf['BPM'].astype('float')
apf['TS%'] = apf['TS%'].astype('float')
apf['3PAr'] = apf['3PAr'].astype('float')
apf['ORB%'] = apf['ORB%'].astype('float')
apf['DRB%'] = apf['DRB%'].astype('float')
apf['TRB%'] = apf['TRB%'].astype('float')
apf['AST%'] = apf['AST%'].astype('float')
apf['STL%'] = apf['STL%'].astype('float')
apf['BLK%'] = apf['BLK%'].astype('float')
apf['TOV%'] = apf['TOV%'].astype('float')
apf['USG%'] = apf['USG%'].astype('float')
apf['OWS'] = apf['OWS'].astype('float')
apf['DWS'] = apf['DWS'].astype('float')
apf['WS'] = apf['WS'].astype('float')
apf['WS/48'] = apf['WS/48'].astype('float')
apf['Age'] = apf['Age'].astype('Int32')
apf['G'] = apf['G'].astype('Int32')
apf['MP'] = apf['MP'].astype('Int32')
apf['PER'] = apf['PER'].astype('float')
apf['FTr'] = apf['FTr'].astype('float')
apf['VORP'] = apf['VORP'].astype('float')

ap_10 = apf.sort_values('MP',ascending=False).groupby('TM').head(10)

#remove asterisk from player name
ap_10['Player'] = ap_10['Player'].str.replace('*', '')

ap_10['YR'] = ap_10['TM'].apply(lambda x : x[-2:])

ap_10['YRprior'] = ap_10['YR'] - 1

ap_10['MPG'] = ap_10['MP'] / ap_10['G']

aprior = ap_10

ap_merged = pd.merge(ap_10, aprior, how='outer', left_on=['Player','YRprior'], right_on = ['Player','YR'])

