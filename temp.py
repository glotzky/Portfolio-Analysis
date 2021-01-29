#pip install scipy
# pip install alpha_vantage
import pandas as pd
import plotly.express as px
from copy import copy
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import plotly.figure_factory as ff
import plotly.graph_objects as go
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.fundamentaldata import FundamentalData
print("Necessary libraries installed sucessfully.")
key = 'FZKZD2AIOWEZM3AR'
ticker = 'MSFT'
ts = TimeSeries(key, output_format='pandas')
#data, meta_data = ts.get_intraday(symbol=ticker,interval='1min', outputsize='full')
#print(data.head(2))
#meta_data
data = ts.get_daily_adjusted(ticker, outputsize='full')
#data
data = data[0]
#data.columns
data = data[['5. adjusted close']]
data = data.rename(columns={'5. adjusted close':'Price'})
#data.head()
data.reset_index(level=0, inplace=True)
data['Price'].round(2)
data = data.rename(columns={'date':'Date'})
#data.head()
data = data.iloc[::-1]
data = data.reset_index(drop=True)
df = data
# Function to plot interactive plot
def interactive_plot(df, title):
    fig = px.line(title=title)
    for i in df.columns[1:]:
        fig.add_scatter(x=df['Date'],y= df[i], name = i)
    fig.show()
# Plot interactive chart
interactive_plot(df,'Price')

#Normalize the data frame
def normalize(df):
    x = df.copy()
    for i in x.columns[1:]:
        x[i]=x[i]/x[i][0]
    return x

interactive_plot(normalize(df),'Normalized Prices')

# TASK #4: PERFORM RANDOM ASSET ALLOCATION AND CALCULATE PORTFOLIO DAILY RETURN
# Let's create random portfolio weights
# Portfolio weights must sum to 1 

# Set random seed
# np.random.seed(101)

# Create random weights for the stocks and normalize them
num=1
weights = np.array(np.random.random(num))
weights

# Ensure that the sum of all weights are = 1
weights = weights/np.sum(weights)
weights
# Normalize the stock avalues 
df_portfolio = normalize(df)
df_portfolio
df_portfolio.columns[1:]
invested_amount = 4000 #amount in dollars ($)
# Note that enumerate returns the value and a counter as well
for counter, stock in enumerate(df_portfolio.columns[1:]):
  df_portfolio[stock] = df_portfolio[stock] * weights[counter]
  df_portfolio[stock] = df_portfolio[stock] * invested_amount #amount of dollars
df_portfolio.round(decimals=2)
# Let's create an additional column that contains the sum of all $ values in the portfolio
df_portfolio['portfolio daily worth in $'] = df_portfolio[df_portfolio != 'Date'].sum(axis = 1)
df_portfolio.round(decimals=2)
# Let's calculate the portfolio daily return 
# Define a new column in the dataframe and set it to zeros
df_portfolio['portfolio daily % return'] = 0.0000

for i in range(1, len(df)):
  # Calculate the percentage of change from the previous day
  df_portfolio['portfolio daily % return'][i] = ( (df_portfolio['portfolio daily worth in $'][i] - df_portfolio['portfolio daily worth in $'][i-1]) / df_portfolio['portfolio daily worth in $'][i-1]) * 100 

df_portfolio.round(decimals = 2)
# TASK #5: PORTFOLIO ALLOCATION - DAILY RETURN/WORTH CALCULATION (FUNCTION)
# Lets assume we have $1,000,000 to be invested and we will allocate this fund based on the weights of the stocks
# We will create a function that takes in the stock prices along with the weights and retun:
# (1) Daily value of each individual securuty in $ over the specified time period
# (2) Overall daily worth of the entire portfolio 
# (3) Daily return 

def portfolio_allocation(df, weights):

  df_portfolio = df.copy()
  
  # Normalize the stock avalues 
  df_portfolio = normalize(df_portfolio)
  
  for counter, stock in enumerate(df_portfolio.columns[1:]):
    df_portfolio[stock] = df_portfolio[stock] * weights[counter]
    df_portfolio[stock] = df_portfolio[stock] * 1000000

  df_portfolio['portfolio daily worth in $'] = df_portfolio[df_portfolio != 'Date'].sum(axis = 1)
  
  df_portfolio['portfolio daily % return'] = 0.0000

  for i in range(1, len(df)):
    
    # Calculate the percentage of change from the previous day
    df_portfolio['portfolio daily % return'][i] = ( (df_portfolio['portfolio daily worth in $'][i] - df_portfolio['portfolio daily worth in $'][i-1]) / df_portfolio['portfolio daily worth in $'][i-1]) * 100 
  
  # set the value of first row to zero, as previous value is not available
  df_portfolio['portfolio daily % return'][0] = 0
  return df_portfolio
  # Call the function
df_portfolio = portfolio_allocation(df, weights)
df_portfolio
# TASK #6: PERFORM PORTFOLIO DATA VISUALIZATION
# Plot the portfolio daily return
fig = px.line(x=df_portfolio.Date, y=df_portfolio['portfolio daily % return'], title = 'Portfolio Daily % Return')
fig.show()
# Plot all stocks (normalized)
interactive_plot(df_portfolio.drop(['portfolio daily worth in $','portfolio daily % return'], axis = 1), 'Portfolio individual stocks worth in $')
# Print out a histogram of daily returns
fig = px.histogram(df_portfolio, x='portfolio daily % return')
fig.show()
fig = px.line(x=df_portfolio.Date,y=df_portfolio['portfolio daily worth in $'],title = 'Portfolio Daily Worth in $')
fig.show()
# TASK #8: CALCULATE PORTFOLIO STATISTICAL METRICS (CUMMULATIVE RETURN, AVERAGE DAILY RETURN, AND SHARPE RATIO)
df_portfolio
# Cummulative return of the portfolio (Note that we now look for the last net worth of the portfolio compared to it's start value)
cummulative_return = ((df_portfolio['portfolio daily worth in $'][-1:] - df_portfolio['portfolio daily worth in $'][0])/ df_portfolio['portfolio daily worth in $'][0]) * 100
print('Cummulative return of the portfolio is {} %'.format(cummulative_return.values[0]))
# Calculate the portfolio standard deviation
print('Standard deviation of portfolio = {}'.format(df_portfolio['portfolio daily % return'].std()))
# Calculate the average daily return 
print('Average daily return of portfolio = {}'.format(df_portfolio['portfolio daily % return'].mean()))
# Portfolio sharpe ratio
sharpe_ratio = df_portfolio['portfolio daily % return'].mean() / df_portfolio['portfolio daily % return'].std() * np.sqrt(252)
print('Sharpe ratio of the portfolio is {}'.format(sharpe_ratio))
# Portfolio Alpha Ratio
print(weights)