import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import plotly.express as px
import pickle
import joblib
import os
default_directory = "E:\RAYHAN SEFAT\Machiene Learn"
os.chdir(default_directory)

with open('Single variable Regression/pickle_model', 'rb') as f:
    mp = pickle.load(f)
print("valule from pickle model {}".format(mp.predict([[3450]])))

mj = joblib.load('Single variable Regression/joblib_model')
print("valule from joblib model {}".format(mj.predict([[3450]])))