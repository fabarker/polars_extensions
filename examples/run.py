import polars as pl
from py_polars_ext import pypolars as pp
import pandas as pd
from py_polars_ext.enums import *
import os
os.environ["POLARS_VERBOSE"] = "1"

path = "/Users/francisbarker/Desktop/PUT_History.csv"
df = pd.read_csv(path, index_col=0)
df.index = pd.to_datetime(df.index)
df = df.reset_index(drop=False)
df['symbol'] = "PUT"
df.columns = ['date','asset_returns','symbol']
returns_df = pl.from_pandas(df)


# Polars Moving Mean Extension
exp_ = pl.col("asset_returns").sliding(window=252)
ewm_ = exp_.ewm(span=5).mean()

print(returns_df.with_columns(ewm_))

print(pp.exponential_weights(125))

