
import pandas as pd
import numpy as np
from process_players import *


a = get_clean_pdata()

touches = pd.read_csv('data/Pstats - touches.csv')

names = {'Otto Porter':'Otto Porter Jr.', 'P.J. Tucker':'PJ Tucker', 'Taurean Waller-Prince':'Taurean Prince',
       'J.R. Smith':'JR Smith', 'J.J. Redick': 'JJ Redick', 'Kelly Oubre': 'Kelly Oubre Jr.', 'Tim Hardaway': 'Tim Hardaway Jr.',
       'Dennis Smith':'Dennis Smith Jr.', 'J.J. Hickson' :'JJ Hickson', 'C.J. Miles':'CJ Miles', 'Nene Hilario': 'Nene',
       'James Ennis':'James Ennis III', 'Larry Nance': 'Larry Nance Jr.', 'Glenn Robinson': 'Glenn Robinson Jr.', 'K.J. McDaniels': 'KJ McDaniels',
       'Wesley Iwundu': 'Wes Iwundu', 'P.J. Hairston':'PJ Hairston', 'Juan Hernangomez':'Juancho Hernangomez'}




names2 = {y:x for x,y in names.items()}

#Keep this one. Changing player names
touches['PLAYER'] = touches['PLAYER'].apply(lambda x : names2[x] if x in names2.keys() else x)

at2 = pd.merge(a, touches[['PLAYER','Year','TOUCHES','PTS PER\nTOUCH' ]], how='left', left_on=['Player','YR'], right_on=['PLAYER','Year']) 

bt2 = at2.dropna()


btg = bt2.sort_values('MP', ascending=False).groupby('TM').head(10)

btgm = btg.groupby('TM').mean()

