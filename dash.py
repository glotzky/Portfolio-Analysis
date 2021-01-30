import streamlit as st
import pandas as pd
import plotly.express as px
from copy import copy
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import plotly.figure_factory as ff
import plotly.graph_objects as go
from alpha_vantage.timeseries import TimeSeries

st.title("Portfolio Dashboard")
st.markdown("This application is a Streamlit dashboard that can be used "
            "to analyze your overall performance over the years")


num_stocks = st.slider('How many stocks are in your portfolio?: ',1,20,1)
invested_amount = st.number_input('What is your principal amount invested?')
#key = 'FZKZD2AIOWEZM3AR'
key = '3JQKJRWTURYJE58J'
tickers = []

def get_tickers(num_stocks):
	for i in range(num_stocks):
		tickers.append(st.text_input("Enter the ticker of Stock %i" %(i+1)))
	return tickers
get_tickers(num_stocks)


ts = TimeSeries(key, output_format='pandas')


#Append new dataframe
def get_tickers_df(tickers):
	port_data = pd.DataFrame({'Price' : [np.nan]})
	for i in range(len(tickers)):
		ticker = tickers[i]
		tic_data = ts.get_daily_adjusted(ticker, outputsize='full')
		tic_data = tic_data[0]
		tic_data = tic_data[['5. adjusted close']]
		tic_data = tic_data.rename(columns={'5. adjusted close': ticker})
		tic_data[ticker].round(2)
		if i == 0:
			port_data = tic_data
		else:
			port_data = pd.concat([port_data, tic_data.reindex(port_data.index)], axis=1)
	return port_data


data = get_tickers_df(tickers)

data.reset_index(level=0, inplace=True)
data = data.rename(columns={'date':'Date'})
st.write(data)
data = data.iloc[::-1]
data = data.reset_index(drop=True)
df = data

st.header("Performance over a 20 year period")
# Function to plot interactive plot
def interactive_plot(df, title):
    fig = px.line(title=title)
    for i in df.columns[1:]:
        fig.add_scatter(x=df['Date'],y= df[i], name = i)
    return fig
# Plot interactive chart
st.write(interactive_plot(df,'Price'))


st.header("Normalize Performance over a 20 year period")
#Normalize the data frame
def normalize(df):
    x = df.copy()
    for i in x.columns[1:]:
        x[i]=x[i]/x[i][0]
    return x

st.write(interactive_plot(normalize(df),'Normalized Prices'))

# TASK #4: PERFORM RANDOM ASSET ALLOCATION AND CALCULATE PORTFOLIO DAILY RETURN
# Portfolio Weights
# Portfolio weights must sum to 1 

# Set random seed
# np.random.seed(101)

st.header('Weights')
option = st.selectbox(
     'Weights:',
     ('Randomized', 'Pre-Determined'))
if option == 'Randomized':
	weights = np.array(np.random.random(num_stocks))
	weights = weights/np.sum(weights)
else:
	weights = []
	for i in range(num_stocks):
		weight = st.number_input("Weight for " + tickers[i])
		#weight = st.number_input("Weight for " + tickers[i]+ " must be less than or equal to 1 and greater than 0: ")
		weights.append(weight)
	weights = weights/np.sum(weights)

# Normalize the stock avalues 


df_portfolio = normalize(df)
df_portfoliot = df_portfolio.columns[1:]

#invested_amount #amount in dollars ($)
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
df_portfolio['portfolio daily ''%'' return'] = 0.0000

for i in range(1, len(df)):
  # Calculate the percentage of change from the previous day
  df_portfolio['portfolio daily ''%'' return'][i] = ( (df_portfolio['portfolio daily worth in $'][i] - df_portfolio['portfolio daily worth in $'][i-1]) / df_portfolio['portfolio daily worth in $'][i-1]) * 100 

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
    df_portfolio[stock] = df_portfolio[stock] * invested_amount

  df_portfolio['portfolio daily worth in $'] = df_portfolio[df_portfolio != 'Date'].sum(axis = 1)
  
  df_portfolio['portfolio daily ''%'' return'] = 0.0000

  for i in range(1, len(df)):
    
    # Calculate the percentage of change from the previous day
    df_portfolio['portfolio daily ''%'' return'][i] = (( (df_portfolio['portfolio daily worth in $'][i] - df_portfolio['portfolio daily worth in $'][i-1]) / df_portfolio['portfolio daily worth in $'][i-1]) * 100 )
  
  # set the value of first row to zero, as previous value is not available
  df_portfolio['portfolio daily ''%'' return'][0] = 0
  return df_portfolio
  # Call the function
df_portfolio = portfolio_allocation(df, weights)
#df_portfolio
# TASK #6: PERFORM PORTFOLIO DATA VISUALIZATION
# Plot the portfolio daily return
st.header("Portfolio Daily Return")
fig = px.line(x=df_portfolio.Date, y=df_portfolio['portfolio daily ''%'' return'], title = 'Portfolio Daily % Return')
st.write(fig)
# Plot all stocks (normalized)
interactive_plot(df_portfolio.drop(['portfolio daily worth in $','portfolio daily ''%'' return'], axis = 1), 'Portfolio individual stocks worth in $')
# Print out a histogram of daily returns
st.header("Histogram of daily returns")
fig = px.histogram(df_portfolio, x='portfolio daily ''%'' return')
st.write(fig)

st.header("Portfolio daily worth in $")
fig = px.line(x=df_portfolio.Date,y=df_portfolio['portfolio daily worth in $'],title = 'Portfolio Daily Worth in $')
st.write(fig)
# TASK #8: CALCULATE PORTFOLIO STATISTICAL METRICS (CUMMULATIVE RETURN, AVERAGE DAILY RETURN, AND SHARPE RATIO)
st.write(df_portfolio)
# Cummulative return of the portfolio (Note that we now look for the last net worth of the portfolio compared to it's start value)
cummulative_return = ((df_portfolio['portfolio daily worth in $'][-1:] - df_portfolio['portfolio daily worth in $'][0])/ df_portfolio['portfolio daily worth in $'][0]) * 100
st.write('Cummulative return of the portfolio is {} %'.format(cummulative_return.values[0]))
# Calculate the portfolio standard deviation
st.write('Standard deviation of portfolio = {}'.format(df_portfolio['portfolio daily ''%'' return'].std()))
# Calculate the average daily return 
st.write('Average daily return of portfolio = {}'.format(df_portfolio['portfolio daily ''%'' return'].mean()))
# Portfolio sharpe ratio
sharpe_ratio = df_portfolio['portfolio daily ''%'' return'].mean() / df_portfolio['portfolio daily ''%'' return'].std() * np.sqrt(252)
st.write('Sharpe ratio of the portfolio is {}'.format(sharpe_ratio))
# Portfolio Alpha Ratio
