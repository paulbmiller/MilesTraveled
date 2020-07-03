# Miles Traveled

## Description
Small project which uses an LSTM to fit the data of [Vehicle Miles Traveled (TRFVOLUSM227NFWA)](https://fred.stlouisfed.org/series/TRFVOLUSM227NFWA) using the monthly data from 1970 to 2018.

## Output
This will output the errors for the different models we decide to run with a graph for visualization.

![Sample graph](graph.png)

## Tunable parameters
We can choose to run multiple combinations by modifying these variables at the end of the script:
- n_input: number of timestamps used for predictions)
- n_preds: number of predictions (size of the test set)
- epochs: number of training epochs
