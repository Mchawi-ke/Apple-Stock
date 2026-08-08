# pip install numpy pandas seaborn matplotlib tensorflow scikit-learn


import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style('whitegrid')
import matplotlib.pyplot as plt
plt.style.use("fivethirtyeight")

import keras
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Dense, LSTM, Dropout ,Bidirectional

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error


df = pd.read_csv("AAPL.csv", parse_dates=True, index_col="Date")
print(df.head())

print(df.tail())

print(df.shape)

print(df.describe())

print(df.info())

print(df.isnull().sum())

print(df.duplicated().sum())

plt.figure(figsize=(15, 6))
df['Open'].plot()
df['Close'].plot()
plt.ylabel(None)
plt.xlabel(None)
plt.title("Opening & Closing Price of Tesla")
plt.legend(['Open Price', 'Close Price'])
plt.tight_layout()
plt.show()


plt.figure(figsize=(15, 6))
df['Volume'].plot()
plt.ylabel('Volume')
plt.xlabel(None)
plt.title("Sales Volume of Apple")
plt.tight_layout()
plt.show()

# We'll use pct_change to find the percent change for each day
plt.figure(figsize=(15, 6))
df['Adj Close'].pct_change().hist(bins=50)
plt.ylabel('Daily Return')
plt.title(f'Apple Dialy Return')
plt.tight_layout()
plt.show()

dataset = df["Close"]
dataset = pd.DataFrame(dataset)

data = dataset.values

print(data.shape)

scaler = MinMaxScaler(feature_range= (0, 1))
scaled_data = scaler.fit_transform(np.array(data).reshape(-1, 1))

print(scaled_data.shape)


# 75% to Train , 25% to Test
train_size = int(len(data)*.75)
test_size = len(data) - train_size

print("Train Size :",train_size,"Test Size :",test_size)

train_data = scaled_data[ :train_size , 0:1 ]
test_data = scaled_data[ train_size-60: , 0:1 ]

print(train_data.shape, test_data.shape)


# Creating a Training set with 60 time-steps and 1 output
x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])


 # Convert to numpy array
x_train, y_train = np.array(x_train), np.array(y_train)


# Reshaping the input
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))


print(x_train.shape , y_train.shape)

base_model1 = keras.layers.Bidirectional(
    keras.layers.LSTM(units=64),  # specify the number of units or any other parameters
    merge_mode="concat"
)

# Define the model structure
model1 = Sequential([
    base_model1,
    Dense(32),
    Dense(16),
    Dense(1)
])

# Compile the model
model1.compile(optimizer='adam', loss='mse', metrics=['mean_absolute_error'])

# Fitting the LSTM to the Training set
callbacks = [EarlyStopping(monitor= 'loss', patience= 10 , restore_best_weights= True)]
history1 = model1.fit(x_train, y_train, epochs= 100, batch_size= 32 , callbacks= callbacks )

# Define the ConvLSTM1D layer with the correct parameters
base_model2 = keras.layers.ConvLSTM1D(
    filters=filters,
    kernel_size=kernel_size,
    strides=1,
    padding="valid",
    activation="tanh",
    recurrent_activation="sigmoid",
    use_bias=True,
    kernel_initializer="glorot_uniform",
    recurrent_initializer="orthogonal",
    bias_initializer="zeros",
    unit_forget_bias=True,
    dropout=0.0,
    recurrent_dropout=0.0,
    return_sequences=False,
    return_state=False,
    go_backwards=False,
    stateful=False,
)

# Add an extra spatial dimension to the input (expanding to 4D)
x_train_expanded = tf.expand_dims(x_train, axis=2)  # Expanding to shape (None, 60, 1, 1)

# Define ConvLSTM1D with the correct input shape
base_model2 = keras.layers.ConvLSTM1D(
    filters=64,  # Number of filters
    kernel_size=1,  # Adjusted kernel size
    strides=1,
    padding="valid",
    activation="tanh",
    recurrent_activation="sigmoid",
    use_bias=True,
    kernel_initializer="glorot_uniform",
    recurrent_initializer="orthogonal",
    bias_initializer="zeros",
    unit_forget_bias=True,
    dropout=0.0,
    recurrent_dropout=0.0,
    return_sequences=False,
    return_state=False,
    go_backwards=False,
    stateful=False,
    input_shape=(x_train_expanded.shape[1], x_train_expanded.shape[2], x_train_expanded.shape[3])  # 4D input shape
)

# Define the model
model2 = Sequential([
    base_model2,
    Dense(32),
    Dense(16),
    Dense(1)
])

# Compile the model
model2.compile(optimizer='adam', loss='mse', metrics=['mean_absolute_error'])

# Early stopping callback
callbacks = [EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)]

# Fit the model with expanded data
history2 = model2.fit(x_train_expanded, y_train, epochs=100, batch_size=32, callbacks=callbacks)





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

