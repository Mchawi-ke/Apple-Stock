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


# Add an extra channel dimension to the input (expanding to 5D)
x_train_expanded = tf.expand_dims(x_train, axis=-1)  # Expanding to shape (None, 60, height, width, 1)

# Define the ConvLSTM2D layer with adjusted parameters
base_model3 = keras.layers.ConvLSTM2D(
    filters=64,  # Define number of filters
    kernel_size=(3, 3),  # Define kernel size
    strides=1,
    padding="same",  # Use 'same' padding to avoid reducing input size
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
    input_shape=(x_train_expanded.shape[1], x_train_expanded.shape[2], x_train_expanded.shape[3], 1)  # 5D input shape
)

# Define the full model
model3 = Sequential([
    base_model3,
    Dense(32),
    Dense(16),
    Dense(1)
])

# Compile the model
model3.compile(optimizer='adam', loss='mse', metrics=['mean_absolute_error'])

# Early stopping callback
callbacks = [EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)]

# Fit the model
history3 = model3.fit(x_train_expanded, y_train, epochs=100, batch_size=32, callbacks=callbacks)




# Sample model history data (Replace these with your actual history data)
# Let's assume history1, history2, and history3 are dictionaries containing loss and accuracy
history1 = {'accuracy': [0.8, 0.85, 0.9], 'loss': [0.5, 0.4, 0.3]}
history2 = {'accuracy': [0.75, 0.82, 0.88], 'loss': [0.6, 0.45, 0.35]}
history3 = {'accuracy': [0.78, 0.83, 0.89], 'loss': [0.55, 0.42, 0.33]}

# Number of epochs
epochs = range(1, len(history1['accuracy']) + 1)

# Plotting Accuracy
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs, history1['accuracy'], label='Model 1 Accuracy', marker='o')
plt.plot(epochs, history2['accuracy'], label='Model 2 Accuracy', marker='o')
plt.plot(epochs, history3['accuracy'], label='Model 3 Accuracy', marker='o')
plt.title('Model Accuracy Comparison')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.xticks(epochs)
plt.ylim(0, 1)
plt.legend()

# Plotting Loss
plt.subplot(1, 2, 2)
plt.plot(epochs, history1['loss'], label='Model 1 Loss', marker='o')
plt.plot(epochs, history2['loss'], label='Model 2 Loss', marker='o')
plt.plot(epochs, history3['loss'], label='Model 3 Loss', marker='o')
plt.title('Model Loss Comparison')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.xticks(epochs)
plt.ylim(0, max(max(history1['loss']), max(history2['loss']), max(history3['loss'])) + 0.1)
plt.legend()

plt.tight_layout()
plt.show()


# Creating a testing set with 60 time-steps and 1 output
x_test = []
y_test = []

for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])
    y_test.append(test_data[i, 0])
x_test, y_test = np.array(x_test), np.array(y_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


#inverse y_test scaling
predictions = model1.predict(x_test)

#inverse predictions scaling
predictions = scaler.inverse_transform(predictions)






fig.add_trace(go.Scatter(x=df.index, y=df["Open"]), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df["Close"]), row=2, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df["High"]), row=3, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df["Low"]), row=4, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df["Adj Close"]), row=5, col=1)
fig.add_trace(go.Scatter(x=df.index, y=df["Volume"]), row=6, col=1)
fig.update_layout(showlegend=False, height=1200, width=800)
fig.show()

