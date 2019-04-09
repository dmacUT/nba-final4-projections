import pandas as pd 

def get_team_target():
    target = pd.read_csv('data/TEAMstats - target_hcw19_2.csv')

    target['TM_x'] = target['TM_x'].str.replace('*', '')
    target['TM_x'] = target['TM_x'] + ' ' + target['YR']

    target['YR'] = target['YR'].str[3:].astype('int32')
    target['YRprior'] = target['YR'] - 1
    target['Team'] = target['TM_x'].str[:-6]

    t = target
    target2 = pd.merge(target, t, how="outer", left_on=["Team", "YRprior"], right_on=["Team", "YR"])
    target2 = target2.drop(['YRprior_x','Team','TM_x_y', 'YR_y', 'YRprior_y'], axis=1)
    target2 = target2[target2['YR_x'] != 5]
    target2['HomeCourt_y'] = target2['HomeCourt_y'].fillna(0)
    target3 = target2.dropna()
    target3 = target3.rename(columns={'TM_x_x':'TM_x'})
    target3 = target3.drop('YR_x', axis=1)

    return target3