# Using ARIMA and Garch model to forcast a timeseries
In this project, the goal is to forcast the stock price of MSFT with the history stock price. Firstly, we do some process to make the time series be stable. Then using auto_arima() function to find the appropriate ARIMA(p,i,q) model. The  residual of the fitting result is autocorrelated. Therefore, we apply garch model to the residual part. Combining two model together, we get our forcasting tool.
