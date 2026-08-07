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

print(df.info())

df["Date"] = pd.to_datetime(df["Date"])

print(df.isnull().sum())

print(df.duplicated().sum())



df = df.set_index("Date")

fig = make_subplots(rows=6, cols=1, 
                    subplot_titles=("Opening Price", "Closing Price", "Highest Price", 
                                    "Lowest Price", "Adjusted Closing Price", "Volume"))

fig.add_trace(go.Scatter(x=df.index, y=df["Open"]), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df["Close"]), row=2, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df["High"]), row=3, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df["Low"]), row=4, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df["Adj Close"]), row=5, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df["Volume"]), row=6, col=1)
fig.update_layout(showlegend=False, height=1200, width=800)
fig.show()

