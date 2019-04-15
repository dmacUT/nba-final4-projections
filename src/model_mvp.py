import pandas as pd
import numpy as np


# 1. Grab mean of all features (BTGM) for all teams
target = pd.read_csv('data/TEAMstats - target.csv')
target['Team'] = target['Team'].str.replace('*', '')
target['TM_x'] = target['Team'] + ' ' + target['YR']

import defvotes

from process_players import *

pdata = get_clean_pdata()
pdata = add_defvotes(pdata, def_votes)

pd = add_2yrs_prior(pdata)

pdmean = get_2yr_mean(pd)

pdmean_10 = pdmean.sort_values('MPmean', ascending=False).groupby('TM_x').head(10)

pdnn = pdmean_10.dropna()

team_m = pdnn.groupby('TM_x').mean()

team_m = team_m.reset_index()

# 2. Calculate ORTG and DRTG based on a few features

# 3. Caculate a NetRTG (O-D)

# 4. Add target to BTGM based on name/yr (to get aligned to teams)

# 5. drop target and make it y variable

# 6. Predict y based off of NetRTG

# 7. OR make X all variables that will add up to ORTG and DRTG


