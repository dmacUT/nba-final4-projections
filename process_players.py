import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing
from team_targets import *
from defvotes import *

def get_clean_pdata():
    allplayers = pd.read_csv('data/adv_pstats_06-18.csv')

    allplayers = allplayers[allplayers.Tm != 'TOT']

    allplayers.drop(['Unnamed: 19', 'Unnamed: 24'], axis=1, inplace=True)

    tconv = pd.read_csv('data/Pstats - teams.csv')

    apf = pd.merge(allplayers, tconv, on='Tm', how='outer')

    apf['TM'] = apf['Team'] + ' ' + apf['YR']

    apf.drop(['Rk', 'Tm','YR', 'Team'], axis=1, inplace=True)

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

    #ap_10 = apf.sort_values('MP',ascending=False).groupby('TM').head(10)
    ap_10 = apf

    #remove asterisk from player name
    ap_10['Player'] = ap_10['Player'].str.replace('*', '')

    ap_10['YR'] = ap_10['TM'].apply(lambda x : x[-2:]).astype('int32')

    ap_10['YRprior'] = ap_10['YR'] - 1

    ap_10['2YRprior'] = ap_10['YR'] - 2

    ap_10['MPG'] = ap_10['MP'] / ap_10['G']

    new = ap_10.sort_values('G', ascending=False).drop_duplicates(subset=['Player', 'YR'], keep='first')

    return new

defvotes = get_defvotes()

def add_defvotes(pdata, defvotes):
    combined = pd.merge(pdata, defvotes, how='outer', on=['Player', 'YR'])
    combined = combined.fillna(value={'advotes':0})
    return combined

def add_2yrs_prior(ap_10):
    aprior = ap_10

    ap = pd.merge(ap_10, aprior, how='outer', left_on=['Player','YRprior'], right_on = ['Player','YR'])

    ap_2yr = pd.merge(ap, aprior, how='outer', left_on=['Player','2YRprior_x'], right_on = ['Player','YR'])

    #ap_2yr = ap_merged.dropna(subset=['Pos_x'])

    #ap_merged = ap_merged[ap_merged['YR_x'] != 6]

    #ap_merged = ap_merged.sort_values('MP_y', ascending=False).groupby('TM_x').head(10)
    return ap_2yr

def get_2yr_mean(df):
    ap_2yr_mean = df[['Player','G_x', 'Pos_x', 'Age_x','TM_x','YR_x','MP_x', 'AgeMulti_x']]

    col_means = ['sPER', 'sTS%', 's3PAr', 'sFTr', 'sORB%',
           'sDRB%', 'sTRB%', 'sAST%', 'sSTL%', 'sBLK%', 'sTOV%', 'sUSG%', 'sOWS', 'sDWS',
           'sWS', 'sWS/48', 'sOBPM', 'sDBPM', 'sBPM', 'sVORP','sMPG', 'sadvotes', 'O_cluster', 'D_cluster','D_clust','O_clust']
    cols = ['sPER_y', 'sTS%_y', 's3PAr_y', 'sFTr_y','sORB%_y','sDRB%_y','sTRB%_y','sAST%_y', 'sSTL%_y', 'sBLK%_y', 'sTOV%_y',
       'sUSG%_y', 'sOWS_y', 'sDWS_y', 'sWS_y', 'sWS/48_y', 'sOBPM_y', 'sDBPM_y', 'sBPM_y', 'sVORP_y', 'sMPG_y',
       'sadvotes_y','O_cluster_y', 'D_cluster_y', 'D_clust_y', 'O_clust_y']

    yr_seven_df = df[df.YR_x == 7][['Player','G_x', 'Pos_x', 'Age_x','TM_x','YR_x','MP_x', 'AgeMulti_x']]

    for pos in ['PF', 'PG', 'SF', 'SG', 'C']:
        for i in range(len(col_means)):
            ap_2yr_mean[pos + str(col_means[i]) + "mn"] = (((df[df['Pos_x'] == pos][col_means[i]]*.8) + (df[df['Pos_x'] == pos][cols[i]]*1.2)) /2) * df['AgeMulti_x']
            yr_seven_df[pos + str(col_means[i]) + "mn"] = df[df['Pos_x'] == pos][cols[i]] * df['AgeMulti_x']
        
    ap_2yr_mean['MPGmean'] = (df['sMPG_y'] + df['sMPG'])/2
    yr_seven_df['MPGmean'] = df['sMPG_y'] 
    
    ap = ap_2yr_mean[ap_2yr_mean['YR_x'] > 7]
    add7 = ap.append(yr_seven_df)

            #new_cols.append(pos + str(col_means[i]) + "mn")
 
    return add7


#combine main player data with d_votes table
pdata = get_clean_pdata()
pdata = add_defvotes(pdata, defvotes)

#add age multiplier
pdata['AgeMulti'] = 1
pdata.loc[pdata['Age'] > 31,'AgeMulti'] = .95
pdata.loc[pdata['Age'] > 34,'AgeMulti'] = .9
pdata.loc[pdata['Age'] < 24, 'AgeMulti'] = 1.2
pdata.loc[pdata['Age'] < 21, 'AgeMulti'] = 1.33

#drop nans and players who played less than 11 games
#p_wage = pdata.dropna()
#p_wage = p_wage[p_wage['G'] > 11]
p_wage = pdata

#Create a list of columns to normalize
cols = ['MP', 'PER', 'TS%', '3PAr', 'FTr', 'ORB%',
       'DRB%', 'TRB%', 'AST%', 'STL%', 'BLK%', 'TOV%', 'USG%', 'OWS', 'DWS',
       'WS', 'WS/48', 'OBPM', 'DBPM', 'BPM', 'VORP','MPG', 'advotes']

#separate p_wage into df's by position
dflist = []
for i in ['PF', 'PG', 'SF', 'SG', 'C']:
    df = p_wage[p_wage['Pos'] == i].reset_index()
    dflist.append(df)

#Make everything a normalized version of itself
dfscaledlist = []
count = 0
for i in ['PF', 'PG', 'SF', 'SG', 'C']:
    x = p_wage[p_wage['Pos'] == i][['MP', 'PER', 'TS%', '3PAr', 'FTr', 'ORB%',
           'DRB%', 'TRB%', 'AST%', 'STL%', 'BLK%', 'TOV%', 'USG%', 'OWS', 'DWS',
           'WS', 'WS/48', 'OBPM', 'DBPM', 'BPM', 'VORP','MPG', 'advotes']] #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled, columns=['sMP', 'sPER', 'sTS%', 's3PAr', 'sFTr', 'sORB%',
           'sDRB%', 'sTRB%', 'sAST%', 'sSTL%', 'sBLK%', 'sTOV%', 'sUSG%','sOWS', 'sDWS',
           'sWS', 'sWS/48', 'sOBPM', 'sDBPM', 'sBPM','sVORP','sMPG', 'sadvotes'])
    dfscaled = pd.concat([dflist[count], df], axis=1, sort=False)
    dfscaledlist.append(dfscaled)
    count += 1

#Rejoin positioned dataframes
dfs = dfscaledlist[0].append(dfscaledlist[1])
dfs = dfs.append(dfscaledlist[2])
dfs = dfs.append(dfscaledlist[3])
dfs = dfs.append(dfscaledlist[4])

#Make offensive and defensive clusters
Xo = dfs[['sPER', 's3PAr','sORB%', 'sAST%', 'sUSG%', 'sOWS','sOBPM', 'sVORP', 'sMPG']]
Xd = dfs[['sDRB%', 'sSTL%', 'sBLK%', 'sDWS', 'sDBPM', 'sVORP','sMPG', 'sadvotes']]

#Create Kmeans models for offense and defense
kmeansO = KMeans(n_clusters=20, random_state=7).fit(Xo)
kmeansD = KMeans(n_clusters=10, random_state=9).fit(Xd)

#Create labels to tie back to individual players
labsO = kmeansO.labels_
labsD = kmeansD.labels_

#Add clusters back to dfs
dfs['O_cluster'] = labsO
dfs['D_cluster'] = labsD

#Create list of defensive indices
dind = list(dfs.groupby('D_cluster').mean().sort_values('DBPM', ascending=False).reset_index()['D_cluster'].values)

#Create list of offensive indices
oind = list(dfs.groupby('O_cluster').mean().sort_values('OBPM', ascending=False).reset_index()['O_cluster'].values)

#Reassign values to numerical values from greatest to least based on highest mean win shares
count = 9
for i in dind:
    dfs.loc[dfs['D_cluster'] == i,'D_clust'] = count
    count -= 1

#Reassign values to numerical values from greatest to least based on highest mean win shares
count = 19
for i in oind:
    dfs.loc[dfs['O_cluster'] == i,'O_clust'] = count
    count -= 1

#Assign position scaled dataframe
p_sca = dfs

#Scale cluster values back to roughly 0-10 (float) scale
p_sca['O_clust'] = (p_sca['O_clust']**1.5)/6
p_sca['D_clust'] = (p_sca['D_clust']**1.5)/3

#Setup player data df to add the prior 2 years of the players' stats
p2yr = p_sca[['Player','Age','Pos','G','MP','TM','YR','YRprior','2YRprior','AgeMulti', 'sPER', 'sTS%', 's3PAr', 'sFTr', 'sORB%',
           'sDRB%', 'sTRB%', 'sAST%', 'sSTL%', 'sBLK%', 'sTOV%', 'sUSG%', 'sOWS', 'sDWS',
           'sWS', 'sWS/48', 'sOBPM', 'sDBPM', 'sBPM', 'sVORP','sMPG', 'sadvotes', 'O_cluster', 'D_cluster','D_clust','O_clust']]

#Create new df of player data with 2 years
p2 = add_2yrs_prior(p2yr)

#Create lists of columns from last year data and 2 years ago data to take the mean of them
lastyr= ['MP_y', 'YR_y', 'YRprior_y', '2YRprior_y', 'AgeMulti_y', 'sPER_y', 'sTS%_y', 's3PAr_y','sFTr_y',
 'sORB%_y', 'sDRB%_y', 'sTRB%_y', 'sAST%_y', 'sSTL%_y', 'sBLK%_y', 'sTOV%_y', 'sUSG%_y', 'sOWS_y', 'sDWS_y',
 'sWS_y', 'sWS/48_y', 'sOBPM_y', 'sDBPM_y', 'sBPM_y', 'sVORP_y', 'sMPG_y', 'sadvotes_y', 'O_cluster_y',
 'D_cluster_y', 'D_clust_y', 'O_clust_y']

twoyrsago = ['MP', 'YR', 'YRprior', '2YRprior', 'AgeMulti', 'sPER', 'sTS%', 's3PAr', 'sFTr',
 'sORB%', 'sDRB%', 'sTRB%', 'sAST%', 'sSTL%', 'sBLK%', 'sTOV%', 'sUSG%', 'sOWS', 'sDWS', 'sWS',
 'sWS/48', 'sOBPM', 'sDBPM', 'sBPM', 'sVORP', 'sMPG', 'sadvotes', 'O_cluster', 'D_cluster', 'D_clust', 'O_clust']

#Copy of df with 2 years data
p3 = p2

#fillna of last year
for i in range(len(lastyr)):
    p3[lastyr[i]] = p3[lastyr[i]].fillna(value=(.75 * p3[twoyrsago[i]]))

#fillna of 2 years ago
for i in range(len(twoyrsago)):
    p3[twoyrsago[i]] = p3[twoyrsago[i]].fillna(value=(p3[twoyrsago[i]]))

#Get mean of last 2 years, and assign it to current year
p2mean = get_2yr_mean(p3)

#Drop rows that don't have a team assigned to it
p3m = p2mean.dropna(subset=['TM_x'])

#Fill NaNs with that position/team data so that it aggregates correctly
p4m = p3m.fillna(p3m.groupby('TM_x').transform('mean'))

#Aggregate players' data into team data by players who played an average of most min. over last 2 years

p2m12 = p4m.sort_values('MPGmean', ascending=False).groupby('TM_x').head(12)
p12 = p2m12.fillna(p2m12.groupby('TM_x').quantile(.3))
tm12 = p12.groupby('TM_x').mean().reset_index()

#Get target of teams making top 4 of conference
team_target = get_team_target()

#Merge aggregated players (not in teams) with targets to lineup training with y
team_df = pd.merge(tm12, team_target, how='left', left_on='TM_x', right_on="TM_x")
team_df_fill0 =team_df.fillna(0)

#drop year 6, since it is beginning of data and has no prior data
team_df_fin = team_df_fill0[team_df_fill0.YR_x !=6]

#drop NaNs
team_fin = team_df_fin.dropna().reset_index()

#Make a list of the team_fin columns to be able to bootstrap training data
team_cols = list(team_fin.columns.values)
team_cols.pop(1)

print('done loading')