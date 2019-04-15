import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import random
import pickle

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error, r2_score, f1_score

from statsmodels.discrete.discrete_model import Logit
from statsmodels.tools import add_constant

team_fin = pd.read_csv('data/5_19_pergame.csv')

def split_time(df, yr_cut = 16):
    final_test_df = df.loc[df['YR_x'] > yr_cut]
    final_test_X = final_test_df.drop('HomeCourt_x', axis=1)
    final_test_y = final_test_df[['TM_x','YR_x','HomeCourt_x']]
    first_train_df = df.loc[df['YR_x'] <= yr_cut]
    return first_train_df, final_test_X, final_test_y

time_train, final_test_X, final_test_y = split_time(team_fin)

X_train = time_train[time_train.YR_x < 14]['PTS_y']

y_train = time_train[time_train.YR_x < 14]['HomeCourt_y']

X_test = time_train[time_train.YR_x > 13]['PTS_y']

y_test = time_train[time_train.YR_x > 13]['HomeCourt_y']

final_tms = final_test_y['TM_x']

X_final = final_test_X['PTS_y']
y_final = final_test_y['HomeCourt_x']

def run_log_reg(Xtr,ytr,Xte,yte):
    #Testing out Logistic Regression
    Xl = Xtr
    X_constl = add_constant(Xl, prepend=True)
    yl = ytr
    log = Logit(yl, X_constl).fit()
    Xte_constl = add_constant(Xte, prepend=True)

    pred_lg = log.predict(exog=Xte_constl)

    mask = pred_lg > .5
    maskint = mask.astype(int)
    print('F1 score: {:.3}'.format(f1_score(yte, maskint)))

    #Roc_Auc score
    print('ROC AUC Score is: {:.3}'.format(roc_auc_score(yte, maskint )))

    tnr, fpr, fnr, tpr = confusion_matrix(yte, maskint).ravel()
    print('Pred True ', tpr, fpr)
    print('Pred False ', fnr, tnr)

    print(log.summary())

    ppg = 'ppg_model.sav'
    pickle.dump(log, open(ppg, 'wb'))

def get_model(model_name):
    loaded_model = pickle.load(open(model_name, 'rb'))
    return loaded_model

#ppg = get_model('ppg_model.sav')


finX_const = add_constant(X_final, prepend=True)
#ppg_probs = ppg.predict(exog=finX_const)
