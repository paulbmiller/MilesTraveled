"""
Module which uses an LSTM to fit the data of Vehicle Miles Traveled
(TRFVOLUSM227NFWA) using the monthly data from 1970 to 2018.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer=100, output_size=1,
                 n_input=12):
        super().__init__()
        self.hidden_layer = hidden_layer
        self.output_size = output_size
        self.n_input = 12
        
        self.lstm = nn.LSTM(input_size, hidden_layer)
        
        self.linear = nn.Linear(hidden_layer*self.n_input, output_size)
        
        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer),
                            torch.zeros(1, 1, self.hidden_layer))
    
    def forward(self, x):
        lstm_out, self.hidden_cell = self.lstm(x, self.hidden_cell)
        predictions = self.linear(lstm_out.reshape(1, -1))
        return predictions



df = pd.read_csv('Miles_Traveled.csv', index_col='DATE')
df.index = pd.to_datetime(df.index)
# in case we want to use statsmodels to gather more info on the data
df.index.freq = 'MS'

df.columns = ['Value']

n_input = 12
n_preds = 12

model = LSTM(n_input=n_input)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

train = df[:-n_preds]
test = df[-n_preds:]

sc = MinMaxScaler()
train_sc = sc.fit_transform(train)
test_sc = sc.transform(test)


inputs = []

for i in range(n_input, len(train)):
    inputs.append((train_sc[i-n_input:i], train_sc[i]))


model.train()
epochs = 20
losses = []

for i in range(epochs):
    for seq, label in inputs:
        optimizer.zero_grad()
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer),
                             torch.zeros(1, 1, model.hidden_layer))
        seq = torch.FloatTensor(seq)
        label = torch.FloatTensor(label).reshape(-1, 1)
        seq = seq.reshape(-1, 1, 1)
        y_pred = model(seq)
        loss = loss_fn(y_pred, label)
        loss.backward()
        optimizer.step()

    L = loss.item()
    losses.append(L)
    print('epoch: {} loss: {:.5f}'.format(i, L))


current_batch = train_sc[-n_input:]
current_batch

model.eval()
preds = []

for i in range(n_preds):
    current_batch = torch.FloatTensor(current_batch).reshape(-1, 1, 1)
    
    with torch.no_grad():
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer),
                    torch.zeros(1, 1, model.hidden_layer))
        pred = model(current_batch).item()
        preds.append(pred)
    
    current_batch = np.append(current_batch[1:], pred)

preds_unsc = sc.inverse_transform(np.array(preds).reshape(-1, 1))

test['Predictions@12'] = preds_unsc.reshape(-1).tolist()

test.plot(figsize=(12, 5))

