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

import prep_model_data


base = return_base_data()

get_final(gb, 0.2)
