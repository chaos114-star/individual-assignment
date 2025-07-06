# %%
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import TensorDataset, DataLoader

# %%
# load data
df = pd.read_csv('data//household_power_consumption.txt', sep = ";")
df.head()

# %%
# check the data
df.info()

# %%
df['datetime'] = pd.to_datetime(df['Date'] + " " + df['Time'])
df.drop(['Date', 'Time'], axis = 1, inplace = True)
# handle missing values
df.dropna(inplace = True)

# %%
print("Start Date: ", df['datetime'].min())
print("End Date: ", df['datetime'].max())

# %%
# split training and test sets
# the prediction and test collections are separated over time
train, test = df.loc[df['datetime'] <= '2009-12-31'], df.loc[df['datetime'] > '2009-12-31']

# %%
# data normalization

scaler = MinMaxScaler()
feature_cols = ['Global_active_power', 'Global_reactive_power', 'Voltage', 
                'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']

train_scaled = train.copy()
test_scaled = test.copy()

train_scaled[feature_cols] = scaler.fit_transform(train[feature_cols])
test_scaled[feature_cols] = scaler.transform(test[feature_cols])

# %%
# split X and y
def create_sequences(data, feature_cols, target_col, seq_length=24):
    X, y = [], []
    values = data[feature_cols + [target_col]].values
    for i in range(len(values) - seq_length):
        X.append(values[i:i+seq_length, :-1])
        y.append(values[i+seq_length, -1])
    return np.array(X), np.array(y)

target_col = 'Global_active_power'
seq_length = 24  # use past 24 hours to predict next hour

X_train, y_train = create_sequences(train_scaled, feature_cols, target_col, seq_length)
X_test, y_test = create_sequences(test_scaled, feature_cols, target_col, seq_length)

# %%
# create dataloaders

batch_size = 64

train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# %%
# build a LSTM model
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # take the last output
        out = self.fc(out)
        return out.squeeze()

input_size = len(feature_cols)
model = LSTMModel(input_size)

# %%
# train the model
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 10

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")

# %%
# evaluate the model on the test set
model.eval()
predictions = []
actuals = []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch)
        predictions.extend(outputs.cpu().numpy())
        actuals.extend(y_batch.numpy())

# %%
# plotting the predictions against the ground truth
import matplotlib.pyplot as plt

plt.figure(figsize=(12,6))
plt.plot(actuals[:500], label='Actual')
plt.plot(predictions[:500], label='Predicted')
plt.xlabel('Time Step')
plt.ylabel('Global_active_power (normalized)')
plt.legend()
plt.title('LSTM Predictions vs Actual')
plt.show()
