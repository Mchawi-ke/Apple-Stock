# pip install numpy pandas matplotlib plotly statsmodels scikit-learn


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error


df = pd.read_csv("AAPL.csv")
print(df.head())

print(df.tail())

print(df.shape)

print(df.describe().T.apply(lambda x: x.apply("{0:.3f}".format)))