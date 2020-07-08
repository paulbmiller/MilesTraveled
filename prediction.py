"""
Module which uses an LSTM to fit the data of Vehicle Miles Traveled
(TRFVOLUSM227NFWA) using the monthly data from 1970 to 2018.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tools.eval_measures import rmse
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from fbprophet import Prophet

import warnings
warnings.filterwarnings("ignore")

class LSTM(nn.Module):
    """Class for the LSTM model"""
    def __init__(self, input_size=1, hidden_layer=100, output_size=1,
                 n_input=12):
        super().__init__()
        self.input_size = input_size
        self.hidden_layer = hidden_layer
        self.output_size = output_size
        self.n_input = n_input
        
        self.lstm = nn.LSTM(input_size, hidden_layer)
        
        self.linear = nn.Linear(hidden_layer * self.n_input, output_size)
        
        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer),
                            torch.zeros(1, 1, self.hidden_layer))
    
    def forward(self, x):
        lstm_out, self.hidden_cell = self.lstm(x, self.hidden_cell)
        predictions = self.linear(lstm_out.reshape(1, -1))
        return predictions


def train_test_split(n_preds, df, scaling=False):
    train = df[:-n_preds]
    
    # Test dataframe as a copy to not get a warning when we insert predictions
    test = df[-n_preds:].copy()

    sc = MinMaxScaler()
    train_sc = sc.fit_transform(train)
    test_sc = sc.transform(test)

    inputs = []

    for i in range(n_input, len(train)):
        inputs.append((train_sc[i-n_input:i], train_sc[i]))
    
    if scaling:
        return train, test, train_sc, test_sc, inputs, sc
    else:
        return train, test, None, None, inputs, None


def lstm(n_input, n_preds, df, epochs):
    """
    This function will create the model, the input sequences, split the deta,
    scale it and train our LSTM model for the given number of epochs.
    It will then predict values for the test set.

    Parameters
    ----------
    n_input : int
        Number of values used for predicting the next value.
    n_preds : int
        Number of predictions we will do (size of the test set).
    df : pd.DataFrame
        Dataset to work on.
    epochs : int
        Number of epochs of training.

    Returns
    -------
    test : pd.DataFrame
        DataFrame of the test split so we can add to it.
    strout : string
        String which qualifies the model (LSTM(args)).
    preds_unsc : list
        Our model's predictions as a list.

    """
    model = LSTM(n_input=n_input)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    train, test, train_sc, test_sc, inputs, sc = train_test_split(n_preds, df,
                                                                  scaling=True)

    model.train()
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
        print('epoch: {} loss: {:.7f}'.format(i, L))
    
    current_batch = train_sc[-n_input:]
    
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
    
    preds_unsc = preds_unsc.reshape(-1).tolist()
    
    strout = 'LSTM ({}, {}, {})'.format(str(n_input), str(n_preds),
                                            str(epochs))
    
    print('Done training and predicting for model ({}, {}, {})'.format(
        n_input, n_preds, epochs))
    
    return test, strout, preds_unsc


def sarima(n_input, n_preds, df):
    
    # To get which SARIMA model to use
    """
    auto_arima(df['Value'], m=12).summary()
    -> SARIMAX(1, 1, 2)x(2, 1, 2, 12)
    """
    train, test, _ , _ , inputs, _ = train_test_split(n_preds, df)
    
    model = SARIMAX(train['Value'], order = (1, 1, 2),
                    seasonal_order=(2, 1, 2, 12))
    results = model.fit()
    
    start=len(train)
    end=len(train)+len(test)-1
    predictions = results.predict(start=start, end=end, dynamic=False,
                                  typ='levels')
    predictions = predictions.tolist()
    
    return test, 'SARIMA ({}, {})'.format( n_input, n_preds), predictions
    


if __name__ == '__main__':
    df = pd.read_csv('Miles_Traveled.csv', index_col='DATE')
    df.index = pd.to_datetime(df.index)
    # in case we want to use statsmodels to gather more info on the data
    df.index.freq = 'MS'

    df.columns = ['Value']

    errors = []

    n_input, n_preds, epochs = 12, 12, 50
    test, strout, preds = lstm(n_input, n_preds, df, epochs)
    test.loc[:, strout] = preds
    errors.append(rmse(test['Value'], test[strout]))
    
    _, strout, preds = sarima(n_input, n_preds, df)
    test.loc[:, strout] = preds
    errors.append(rmse(test['Value'], test[strout]))
    
    # Since the prophet model is so specific, we will work with it alone
    df_prophet = pd.read_csv('Miles_Traveled.csv')
    df_prophet.columns = ['ds', 'y']
    df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])
    
    train = df_prophet.iloc[:-n_preds]
    
    strout = 'Prophet'
    m = Prophet()
    m.fit(train)
    future = m.make_future_dataframe(periods=12, freq='MS')
    forecast = m.predict(future)
    
    test[strout] = forecast['yhat'].iloc[-n_preds:].values
    errors.append(rmse(test['Value'], test[strout]))
    
    # To try multiple combinations for the LSTM, copy these lines and modify
    # the 3 variables
    """
    n_input, n_preds, epochs = 12, 12, 20
    temp_test, strout, preds = sarima(n_input, n_preds, df)
    if len(temp_test) > len(test):
        test = pd.concat([test, temp_test.iloc[:-len(test)]])
    test.loc[temp_test.index, strout] = preds
    errors.append(rmse(test['Value'], test[strout]))
    """
    
    test.plot(figsize=(12, 8), title='Vehicle Miles Traveled')

    print('RMSE errors: {}'.format(errors))

