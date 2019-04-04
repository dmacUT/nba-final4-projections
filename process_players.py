import pandas as pd
import numpy as np

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
    return ap_10
#####

def add_oimp(ap_10):
    #attempt at own offensive impact metric
    ap_10['Oimp'] = ((ap_10.MPG*1.2) * ((ap_10['USG%']/100)*1.1) * (1+(np.exp((ap_10['TS%']-ap_10['TS%'].mean())/np.sqrt(ap_10['TS%'].std()))))*.5) + (ap_10['3PAr'] * 0.125 *ap_10.MPG) + ((ap_10['AST%']/100) * ap_10.MPG * 0.1)
    return ap_10

def add_adj_dbpm(ap_10):
    #attempt at own defensive metric
    ap_10['adj_DBPM'] = np.exp(ap_10.DBPM) * (ap_10.MPG / 48)
    return ap_10

def add_defvotes(pdata, defvotes):
    combined = pd.merge(pdata, defvotes, how='outer', on=['Player', 'YR'])
    combined = combined.fillna(value={'advotes':0})
    return combined


def add_2yrs_prior(ap_10):
    aprior = ap_10

    #2nd attempt at offensive impact

    #####

    ap = pd.merge(ap_10, aprior, how='outer', left_on=['Player','YRprior'], right_on = ['Player','YR'])

    ap_2yr = pd.merge(ap, aprior, how='outer', left_on=['Player','2YRprior_x'], right_on = ['Player','YR'])

    #ap_2yr = ap_merged.dropna(subset=['Pos_x'])

    #ap_merged = ap_merged[ap_merged['YR_x'] != 6]

    #ap_merged = ap_merged.sort_values('MP_y', ascending=False).groupby('TM_x').head(10)
    return ap_2yr

def get_2yr_mean(df):
    ap_2yr_mean = df[['Player', 'Pos_x', 'Age_x','TM_x','YR_x','MP_x', 'AgeMulti_x']]

    col_means = ['sPER', 'sTS%', 's3PAr', 'sFTr', 'sORB%',
           'sDRB%', 'sTRB%', 'sAST%', 'sSTL%', 'sBLK%', 'sTOV%', 'sUSG%', 'sOWS', 'sDWS',
           'sWS', 'sWS/48', 'sOBPM', 'sDBPM', 'sBPM', 'sVORP','sMPG', 'sadvotes', 'O_cluster', 'D_cluster','D_clust','O_clust']
    print(len(col_means))
    cols = ['sPER_y', 'sTS%_y', 's3PAr_y', 'sFTr_y','sORB%_y','sDRB%_y','sTRB%_y','sAST%_y', 'sSTL%_y', 'sBLK%_y', 'sTOV%_y',
       'sUSG%_y', 'sOWS_y', 'sDWS_y', 'sWS_y', 'sWS/48_y', 'sOBPM_y', 'sDBPM_y', 'sBPM_y', 'sVORP_y', 'sMPG_y',
       'sadvotes_y','O_cluster_y', 'D_cluster_y', 'D_clust_y', 'O_clust_y']
    print(len(cols))

    for i in range(len(col_means)):
        ap_2yr_mean[str(col_means[i])+"mean"] = (df[col_means[i]] + df[cols[i]])/2
    
    return ap_2yr_mean


