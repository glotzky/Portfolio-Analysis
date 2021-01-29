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
data = ts.get_daily_adjusted(ticker)
#data
data = data[0]
#data.columns
data = data[['5. adjusted close']]
data.rename(columns={'5. adjusted close':'Price'})
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
