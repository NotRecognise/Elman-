#!/usr/bin/env python
# coding: utf-8

# In[3]:


pip install torch



# In[4]:


conda install pytorch torchvision torchaudio cpuonly -c pytorch


# In[5]:


import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# In[13]:


data=pd.read_csv(r'C:\Users\lenovo\OneDrive\Desktop\yahoo_stock.csv')


# In[15]:


data.head()


# In[14]:


data.tail()


# In[16]:


data.info()


# In[17]:


data.dtypes


# In[18]:


data = data[['Close']]


# In[19]:


scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)
scaler.fit(data[['Close']])


# In[21]:


def create_sequences(data, sequence_length):
    sequences = []
    targets = []
    for i in range(len(data) - sequence_length):
        sequences.append(data[i:i+sequence_length])
        targets.append(data[i+sequence_length])
    return np.array(sequences), np.array(targets)


# In[22]:


sequence_length = 10  
X, y = create_sequences(data_scaled, sequence_length)


# In[23]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[24]:


X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)


# In[25]:


class ElmanNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ElmanNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.input_to_hidden = nn.Linear(input_size + hidden_size, hidden_size)
        self.hidden_to_output = nn.Linear(hidden_size, output_size)
        self.activation = nn.Tanh()

    def forward(self, x, hidden):
        outputs = []
        for t in range(x.size(1)):  
            combined = torch.cat((x[:, t, :], hidden), dim=1)
            hidden = self.activation(self.input_to_hidden(combined))
            output = self.hidden_to_output(hidden)
            outputs.append(output)
        return torch.stack(outputs, dim=1), hidden


# In[26]:


input_size = 1  
hidden_size = 50
output_size = 1
model = ElmanNetwork(input_size, hidden_size, output_size)


# In[27]:


criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# In[28]:


class ElmanNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ElmanNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.input_to_hidden = nn.Linear(input_size + hidden_size, hidden_size)
        self.hidden_to_output = nn.Linear(hidden_size, output_size)
        self.activation = nn.Tanh()

    def forward(self, x, hidden):
        batch_size = x.size(0)  # Get batch size
        if hidden.size(0) != batch_size:  # Reset hidden if batch size changes
            hidden = hidden.expand(batch_size, -1).contiguous()

        outputs = []
        for t in range(x.size(1)):  # Loop through time steps
            combined = torch.cat((x[:, t, :], hidden), dim=1)
            hidden = self.activation(self.input_to_hidden(combined))
            output = self.hidden_to_output(hidden)
            outputs.append(output)
        return torch.stack(outputs, dim=1), hidden


# In[29]:


epochs = 100
hidden = torch.zeros(1, hidden_size)  
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    hidden = torch.zeros(X_train.size(0), hidden_size)

    output, hidden = model(X_train, hidden)
    loss = criterion(output[:, -1], y_train)

    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")


# In[30]:


model.eval()
with torch.no_grad():
    hidden = torch.zeros(X_test.size(0), hidden_size)

    predictions, _ = model(X_test, hidden)
    predictions = predictions[:, -1].squeeze()  

    y_test_unscaled = scaler.inverse_transform(y_test.numpy().reshape(-1, 1))
    predictions_unscaled = scaler.inverse_transform(predictions.numpy().reshape(-1, 1))

plt.figure(figsize=(10, 6))
plt.plot(y_test_unscaled, label="Actual Prices", color="blue")
plt.plot(predictions_unscaled, label="Predicted Prices", color="orange")
plt.legend()
plt.title("Stock Price Prediction")
plt.xlabel("Days")
plt.ylabel("Price")
plt.show()


# In[31]:


print(f"Shape of X_test: {X_test.shape}") 


# In[32]:


print(f"Shape of hidden: {hidden.shape}")  


# In[33]:


full_data_scaled = scaler.transform(data[['Close']])

X_full, _ = create_sequences(full_data_scaled, sequence_length)

X_full = torch.tensor(X_full, dtype=torch.float32)


# In[34]:


hidden = torch.zeros(X_full.size(0), hidden_size)

model.eval()
with torch.no_grad():
    predictions, _ = model(X_full, hidden)
    predictions = predictions[:, -1].squeeze() 
predictions_unscaled = scaler.inverse_transform(predictions.numpy().reshape(-1, 1))


# In[38]:


predicted_data = pd.DataFrame({  
    "Actual": data["Close"].iloc[sequence_length:].reset_index(drop=True),
    "Predicted": predictions_unscaled.flatten()
})

print(predicted_data.head())


# In[42]:


data.columns = data.columns.str.strip()


# In[43]:


print(len(data))
print(sequence_length)


# In[44]:


data["Date"] = pd.date_range(start="2023-01-01", periods=len(data), freq='D')


# In[45]:


print(data["Date"].iloc[sequence_length:])  
print(data["Close"].iloc[sequence_length:])  
print(predictions_unscaled.flatten())  


# In[48]:


import matplotlib.pyplot as plt

# Strip column names of extra spaces
predicted_data.columns = predicted_data.columns.str.strip()

# Check if 'Date' exists
if 'Date' in predicted_data.columns:
    plt.figure(figsize=(12, 6))
    plt.plot(predicted_data["Date"], predicted_data["Actual"], label="Actual Prices", color="blue")
    plt.plot(predicted_data["Date"], predicted_data["Predicted"], label="Predicted Prices", color="orange")
    plt.xticks(rotation=45)
    plt.legend()
    plt.show()
else:
    print("The 'Date' column does not exist.")


# In[49]:


predicted_data.to_csv("predicted_stock_prices.csv", index=False)

