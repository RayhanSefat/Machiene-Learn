import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import plotly.express as px
import pickle
import os
default_directory = "E:\RAYHAN SEFAT\Machiene Learn"
os.chdir(default_directory)

with open('Single variable Regression/pickle_model', 'rb') as f:
    md = pickle.load(f)
    print(md.predict([[3450]]))