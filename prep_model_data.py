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

team_fin = pd.read_csv('data/5_19_aggdataWS.csv')

#Make a list of the team_fin columns to be able to bootstrap training data
team_cols = list(team_fin.columns.values)
team_cols.pop(1)

team_base = pd.read_csv('data/5_19_pergame.csv')


def bootstrap(arr, iterations=10):
    """Create a series of bootstrapped samples of an input array.
    Parameters
    ----------
    arr: Numpy array
        1-d numeric data
    iterations: int, optional (default=10000)
        Number of bootstrapped samples to create.
    Returns
    -------
    boot_samples: list of arrays
        A list of length iterations, each element is array of size of input arr
    """
    cols = list(arr.columns.values)
    
    if type(arr) != np.ndarray:
        arr = np.array(arr)

    if len(arr.shape) < 2:
        arr = arr[:, np.newaxis]
        # [:, np.newaxis] increases the dimension of arr from 1 to 2

    nrows = arr.shape[0]
    #boot_samples = []
    df_list = []
    
    for _ in range(iterations):
        row_inds = np.random.randint(nrows, size=nrows)
        # because of the [:, np.newaxis] above 
        # the following will is a 1-d numeric data with the same size as the input arr
        boot_sample = arr[row_inds, :]
        
        #Rejoin positioned dataframes
        dfs = pd.DataFrame(boot_sample, columns=cols)
        df_list.append(dfs)
        
    df_merged = pd.concat(df_list)
    colsnoTM_x = cols
    colsnoTM_x.pop(1)
    df_merged[colsnoTM_x] = df_merged[colsnoTM_x].apply(pd.to_numeric)

    return df_merged

def split_time(df, yr_cut = 18):
    #randlist has a random index for % length of index
    #last3yrs = df.sort_values('YR_x', ascending = False).reset_index()
    #final_test_df = last3yrs.iloc[:(int(len(df) * test_size))]
    final_test_df = df.loc[df['YR_x'] > yr_cut]
    final_test_X = final_test_df.drop('HomeCourt_x', axis=1)
    final_test_y = final_test_df[['TM_x','YR_x','HomeCourt_x']]
    first_train_df = df.loc[df['YR_x'] <= yr_cut]
    return first_train_df, final_test_X, final_test_y

time_train, final_test_X, final_test_y = split_time(team_fin)

tt_shrunk = time_train[['YR_x', 'PFsVORPmn', 'PGsVORPmn', 'SFsVORPmn', 'CsVORPmn', 'SGsVORPmn','PFsadvotesmn','PFsOWSmn','PFsDWSmn',
'SFsadvotesmn','SFsOWSmn','SFsDWSmn','Csadvotesmn','CsOWSmn','CsDWSmn','SGsadvotesmn','SGsOWSmn','SGsDWSmn','PGsadvotesmn','PGsOWSmn','PGsDWSmn',
 'HomeCourt_y', 'HomeCourt_x',
 'CO_SSmn', 'CD_SSmn', 'PFO_SSmn', 'PFD_SSmn', 'PGO_SSmn', 'PGD_SSmn', 'SGO_SSmn', 'SGD_SSmn' , 'SFO_SSmn', 'SFD_SSmn'
 ]]


#Create bootstrapped sample of TIME main_training data set
tt_boot = bootstrap(tt_shrunk, iterations=15)

tt_X = tt_boot[[
'PFsVORPmn', 'PGsVORPmn', 'SFsVORPmn', 'CsVORPmn', 'SGsVORPmn','PFsadvotesmn','PFsOWSmn','PFsDWSmn',
'SFsadvotesmn','SFsOWSmn','SFsDWSmn','Csadvotesmn','CsOWSmn','CsDWSmn','SGsadvotesmn','SGsOWSmn','SGsDWSmn','PGsadvotesmn','PGsOWSmn','PGsDWSmn',
'CO_SSmn', 'CD_SSmn', 'PFO_SSmn', 'PFD_SSmn', 'PGO_SSmn', 'PGD_SSmn', 'SGO_SSmn', 'SGD_SSmn' , 'SFO_SSmn', 'SFD_SSmn']]

tt_y = tt_boot['HomeCourt_x']

tt_Xtest = final_test_X[final_test_X.YR_x < 17][[
    'PFsVORPmn', 'PGsVORPmn', 'SFsVORPmn', 'CsVORPmn', 'SGsVORPmn','PFsadvotesmn','PFsOWSmn','PFsDWSmn',
'SFsadvotesmn','SFsOWSmn','SFsDWSmn','Csadvotesmn','CsOWSmn','CsDWSmn','SGsadvotesmn','SGsOWSmn','SGsDWSmn','PGsadvotesmn','PGsOWSmn','PGsDWSmn',
'CO_SSmn', 'CD_SSmn', 'PFO_SSmn', 'PFD_SSmn', 'PGO_SSmn', 'PGD_SSmn', 'SGO_SSmn', 'SGD_SSmn' , 'SFO_SSmn', 'SFD_SSmn'
 ]]
tt_ytest = final_test_y[final_test_y.YR_x < 17]['HomeCourt_x']

final_X = final_test_X[final_test_X.YR_x > 18][[
    'PFsVORPmn', 'PGsVORPmn', 'SFsVORPmn', 'CsVORPmn', 'SGsVORPmn','PFsadvotesmn','PFsOWSmn','PFsDWSmn',
'SFsadvotesmn','SFsOWSmn','SFsDWSmn','Csadvotesmn','CsOWSmn','CsDWSmn','SGsadvotesmn','SGsOWSmn','SGsDWSmn','PGsadvotesmn','PGsOWSmn','PGsDWSmn',
'CO_SSmn', 'CD_SSmn', 'PFO_SSmn', 'PFD_SSmn', 'PGO_SSmn', 'PGD_SSmn', 'SGO_SSmn', 'SGD_SSmn' , 'SFO_SSmn', 'SFD_SSmn'
 ]]
final_y = final_test_y[final_test_y.YR_x > 18]['HomeCourt_x']

final_tms = final_test_y[final_test_y.YR_x > 18]['TM_x']

# Base model data
team_base = pd.read_csv('data/5_19_pergame.csv')

base_train, base_test_X, base_test_y = split_time(team_base)

X_train = base_train['PTS_y']

y_train = base_train['HomeCourt_x']

X_final_b = base_test_X[base_test_X.YR_x > 18]['PTS_y']

y_final_b = base_test_y[base_test_y.YR_x > 18]['HomeCourt_x']

base_tms = base_test_y ['TM_x']



def run_rf_model(Xtr,ytr,Xte,yte, n_est = 500, thresh=0.5):
    rf2 = RandomForestClassifier(bootstrap=True, n_estimators=n_est,
                           max_features='auto',
                           random_state=random.randint(0,1000), n_jobs=1)
    rf2.fit(Xtr, ytr)

    #F1 score
    pred_rf = rf2.predict(Xte)
    print('F1 score: {:.3}'.format(f1_score(yte, pred_rf)))

    #Roc_Auc score
    print('ROC AUC Score is: {:.3}'.format(roc_auc_score(yte, rf2.predict_proba(Xte).T[1] )))

    rf_probs = rf2.predict_proba(Xte).T[1]

    pp = rf_probs > thresh
    pp = pp.astype(int)

    tnr, fpr, fnr, tpr = confusion_matrix(yte, pp).ravel()

    print('Pred True ', tpr, fpr)
    print('Pred False ', fnr, tnr)

    fpr, tpr, thresholds = roc_curve(yte, rf_probs)
    plt.figure(figsize=(5,5))
    plt.plot( fpr, tpr )#, c=thresholds/thresholds.max(), cmap="viridis" )
    plt.plot( [0,1],[0,1], "--" )
    plt.title( "ROC Curve" )
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")

    rf_model1 = 'rf_model18.sav'
    pickle.dump(rf2, open(rf_model1, 'wb'))
    

def get_model(model_name):
    loaded_model = pickle.load(open(model_name, 'rb'))
    return loaded_model
    

def run_gb_model(Xtr,ytr,Xte,yte, lr = 0.1, thresh=0.5):
    gmodel = GradientBoostingClassifier(learning_rate=lr, random_state=random.randint(0,1000))
    gmodel.fit(Xtr, ytr)

    #F1 score
    pred_gb = gmodel.predict(Xte)
    print('F1 score: {:.3}'.format(f1_score(yte, pred_gb)))

    #Roc_Auc score
    print('ROC AUC Score is: {:.3}'.format(roc_auc_score(yte, gmodel.predict_proba(Xte).T[1] )))

    gb_probs = gmodel.predict_proba(Xte).T[1]

    gg = gb_probs > thresh
    gg = gg.astype(int)

    tnr, fpr, fnr, tpr = confusion_matrix(yte, gg).ravel()

    print('Pred True ', tpr, fpr)
    print('Pred False ', fnr, tnr)

    fpr, tpr, thresholds = roc_curve(yte, gb_probs)
    plt.figure(figsize=(5,5))
    plt.plot( fpr, tpr )#, c=thresholds/thresholds.max(), cmap="viridis" )
    plt.plot( [0,1],[0,1], "--" )
    plt.title( "ROC Curve" )
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")

    gb_model18 = 'gb_model18.sav'
    pickle.dump(gmodel, open(gb_model18, 'wb'))

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

    base = 'log18_model.sav'
    pickle.dump(log, open(base, 'wb'))

# rf2 = get_model('rf_model2.sav')
# gb2 = get_model('gb_model2.sav')
ppg = get_model('ppg_model.sav')

finX = X_final_b
finX_const = add_constant(finX, prepend=True)
ppg_probs = ppg.predict(exog=finX_const)



def get_final(model, thresh):
    #FINAL TRY score
    pred_fin = model.predict(final_X)
    fin_probs = model.predict_proba(final_X).T[1]
    fp = fin_probs > thresh
    fp = fp.astype(int)
    bp = ppg_probs > 0.23
    bp = bp.astype(int)

    print('dmacUT model F1 score: {:.3}'.format(f1_score(final_y, fp)))
    print(' ')
    print('Base model F1 score: {:.3}'.format(f1_score(y_final_b, bp)))
    print(' ')

    #fin_probs = model.predict_proba(final_X2).T[1]
    #Roc_Auc score
    #print('ROC AUC Score is: {:.3}'.format(roc_auc_score(final_y, fin_probs )))

    tnr, fpr, fnr, tpr = confusion_matrix(final_y, fp).ravel()
    print('dmacUT model confusion matrix:')
    print('Pred True ', tpr, fpr)
    print('Pred False ', fnr, tnr)
    print(' ')

    btnr, bfpr, bfnr, btpr = confusion_matrix(y_final_b, bp).ravel()
    print('base model confusion matrix:')
    print('Pred True ', btpr, bfpr)
    print('Pred False ', bfnr, btnr)
    print(' ')

    tmact = np.array([np.array(final_tms), np.array(final_y)])
    tmact_df = pd.DataFrame({'Teams':tmact[0],'Finish Top 8':tmact[1]})

    tmprobs = np.array([np.array(final_tms), fin_probs, ppg_probs])

    tmprobs_df = pd.DataFrame({'Teams':tmprobs[0],'dmacUTProbs':tmprobs[1], 'BaseProbs':tmprobs[2]})
    joined = pd.merge(tmact_df, tmprobs_df, how='left', on='Teams')
    joined['YR'] = joined.Teams.str[-2:].astype(int)
    sort_tmprobs = joined.sort_values(['YR','dmacUTProbs'], ascending = False).reset_index()
    sort_tmprobs = sort_tmprobs.drop('index', axis=1)
    df19 = sort_tmprobs[sort_tmprobs['YR'] == 19]
    df18 = sort_tmprobs[sort_tmprobs['YR'] == 18]
    df17 = sort_tmprobs[sort_tmprobs['YR'] == 17]
    #return sort_tmprobs
    print(df19)
    print(' ')
    print(df18)
    print(' ')
    print(df17)


#get_final(gbSS, 0.17)