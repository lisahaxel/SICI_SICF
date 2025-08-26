#%% Dependencies for helper_functions.py and Machine_Learning.py
"""""
This script contains the code to load all the relevant libraries for Machine_Learning.py. 
"""
# Libraries
import os
import sys
import re
import warnings
import time
import pickle
import random
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm 
from joblib import Parallel, delayed
import concurrent.futures 
from datetime import datetime
import json 
import numpy as np 
import matplotlib.pyplot as plt 
import scipy 
import scipy.io
import sklearn 
import pandas as pd 
import numpy as np
import seaborn as sns 
import mlxtend 
from joblib import Parallel, delayed
from scipy.stats import ks_2samp
from src.mRMR_feature_select import mRMR_feature_select
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import learning_curve
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFECV
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
# %%
