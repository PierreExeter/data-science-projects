import pygwalker as pyg
import pandas as pd


df = pd.read_csv("https://kanaries-app.s3.ap-northeast-1.amazonaws.com/public-datasets/bike_sharing_dc.csv", parse_dates=['date'])

pyg.walk(df)
